import math
import collections
from torch.utils.data import Dataset, DataLoader
from respr.data.base import BaseDataAdapter
from respr.data.augmentation import DATA_AUG_FACTORY
from respr.util.common import fill_missing_values, BaseFactory
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from loguru import logger
import copy
import random
import scipy.signal
from collections import defaultdict
import tqdm
from pathlib import Path

DTYPE_FLOAT = np.float32

# TODO: set seeds for random 

def create_train_val_test_split(sample_ids: list, val=0.2, test=0.2):
    # the _junk suffix means that we drop that variable completely
    train_samples, val_samples = train_test_split(sample_ids, test_size=val)

    adjusted_fraction = test/(1 - val)
    train_samples, test_samples = \
        train_test_split(train_samples, test_size=adjusted_fraction) 
    
    return train_samples, val_samples, test_samples

class ResprStandardIndexedDataContainer:
    """ 
    Interface to access the dataset file created by `IndexedDatasetBuilder`
    This container is supposed to be shared across multiple indexed `Dataset` 
    (torch.utils.data.Dataset). In such dataset only part of the index will be 
    available and data will be sourced for this container.
    """
    
    def __init__(self, config) -> None:
        self._config = config
        self.dataset_file_path = self._config["dataset_file_path"]
        self.indexed_data = self.load_data()
    
    def load_data(self):
        import pickle
        with open(self.dataset_file_path, "rb") as f:
            data = pickle.load(f)
        return data
    
    def __add__(self, other):
        """Merges this container with the `other` and returns `self`.
        In case of dataset_id conflict, the dataset_id in the `other` is
        suffixed with integer (starting from 1 and increasing by 1) 
        until a unique id is found.
        """
       
        self.indexed_data = self._fuse_metadata(other)
        self.indexed_data, d2_id_to_new_id_map = \
            self._resolve_dataset_ids_and_indices(other)
        
        # Translate and merge incoming datasets
        self.indexed_data = self._merge_datasets(other, d2_id_to_new_id_map)
        
        # Translate and merge the incoming indices 
        self.indexed_data = self._merge_indices(other, d2_id_to_new_id_map)
        return self

    def _merge_indices(self, other, other_id_to_new_id_map):
        idx1 = self.indexed_data["index"]
        idx2 = other.indexed_data["index"]
        
        for i, item in enumerate(idx2):
            old_dataset_idx, sample_id, offset = item
            old_datset_id = other.indexed_data['dataset_index_to_id'][
                old_dataset_idx]
            new_dataset_id = other_id_to_new_id_map[old_datset_id]
            new_dataset_idx = self.indexed_data['dataset_id_to_index'][
                new_dataset_id
            ]
            
            translated = (new_dataset_idx, sample_id, offset)
            idx1.append(translated)
            
        return self.indexed_data
        
        
        
    def _merge_datasets(self, other, other_id_to_new_id_map):
        d1 = self.indexed_data
        m = other_id_to_new_id_map
        
        for old_id in m:
            new_id = m[old_id]
            new_entry = {
                'dataset_id': new_id,
                'sample_ids': other.indexed_data['datasets'][old_id]['sample_ids'],
                'samples': other.indexed_data['datasets'][old_id]['samples']
                #>>> old: 'y': other.indexed_data['datasets'][old_id]['y']
            }
            d1['datasets'][new_id] =  new_entry
        
        return self.indexed_data
        
    
    def _fuse_metadata(self, other):
        """Correctly merge old _metadata and create new one. At 
        present just checks if vector length matches"""
        m1 = self.indexed_data["_metadata"]
        m2 = other.indexed_data["_metadata"]
        
        assert m1["vector_length"] == m2["vector_length"]
        return self.indexed_data
        
    def _resolve_dataset_ids_and_indices(self, other):
        """Find unique dataset_id in case of id conflict and 
        indices for the incoming datasets (from the `other`)"""
        d1 = self.indexed_data
        d2 = other.indexed_data
        existing_ids = d1['dataset_id_to_index']
        
        d2_id_to_new_id_map = {}
        new_id_to_d2_id_map = {}
        
        for id_ in  d2['dataset_id_to_index']:
            new_id = id_
            if (id_ in existing_ids) or \
                (id_ in new_id_to_d2_id_map):
                new_id = self._generate_new_id(id_, existing_ids,
                                               new_id_to_d2_id_map)
            d2_id_to_new_id_map[id_] = new_id
            new_id_to_d2_id_map[new_id] = id_
            
        self._sanity_check_index_to_id_maps(d1['dataset_index_to_id'],
                                            d1['dataset_id_to_index'])
        self._sanity_check_index_to_id_maps(d2['dataset_index_to_id'],
                                            d2['dataset_id_to_index'])
        
        current_max_idx = -1
        
        for k in d1['dataset_index_to_id']:
            current_max_idx = max(k, current_max_idx)
        
        

        
        current_idx = current_max_idx
        for dataset_id in sorted(new_id_to_d2_id_map.keys()):
            current_idx += 1
            assert dataset_id not in d1['dataset_id_to_index']
            d1['dataset_id_to_index'][dataset_id] = current_idx
            assert current_idx not in d1['dataset_index_to_id']
            d1['dataset_index_to_id'][current_idx] = dataset_id
            
        return self.indexed_data, d2_id_to_new_id_map
        
        
        
            
    def _generate_new_id(self, id_, m1, m2):
        start = 0
        while True:
            start += 1
            suffix = str(start).zfill(4)
            new_id = f"{id_}_{suffix}"
            if not ((new_id in m1) or (new_id in m2)):
                return new_id
    
    def _sanity_check_index_to_id_maps(self, idx_to_id, id_to_idx):
        assert len(idx_to_id) == len(id_to_idx)
        for k in idx_to_id:
            assert k == id_to_idx[idx_to_id[k]]
        
        
        
        
        
        
        
    
    
class _ResprDataset(Dataset):
    def __init__(self, config, datasource: BaseDataAdapter, sample_ids: list) -> None:
        """_summary_
        #TODO: complete doc string
        Args:
            config (_type_): _description_
            datasource (BaseDataAdapter): _description_
            sample_ids (list): list of sample ids to use
        """
        super().__init__()
        self._config = config
        self.datasource = datasource
        self.num_samples = len(sample_ids)
        self.sample_ids = sample_ids
        self.window_duration = 32
        self.window_step_duration = 1
        self.expected_feature_size = None
        self.cache = {}
        self.use_cache = True
        
    def initialize_context(self):
        
        counters = []
        for id_ in sample_ids:
            counters.append(
                {   "id": id_,
                    "i": 0,
                    "valid_patch": None
                }
            )
        
        self.context = {
            "counters" : counters
        }
        
    
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        pass
    
    def get_next_window_info(self, index: int):
        pass
        


class BaseResprCsvDataset(Dataset):
    def __init__(self, config={}, datasource=None) -> None:
        super().__init__()
        self._config = self.prepare_config(config)
        self.num_samples = None
        if datasource is None:
            self._dataset_path = self._config["dataset_path"]
            self.initialize_source()
        else:
            self.set_source(datasource)

    def prepare_config(self, config):
        self._config = config
        return self._config
    
    def initialize_source(self):
        """Read the whole dataset or do setup to read the data from disk on
        demand. Also do an initial scan to setup size (`num_samples`) and
        index for index base access."""
        df = pd.read_csv(self._dataset_path)
        self.set_source(df)

    def set_source(self, df):
        cols = df.keys()
        self.x_cols = sorted([c for c in cols if c.startswith("x_")])
        x_cols_idx = [int(c.split("x_")[1]) for c in self.x_cols]
        x_idx_0 = x_cols_idx[0]
        assert x_idx_0 == 0
        x_idx_max = x_cols_idx[-1]
        assert (x_idx_max + 1) == len(x_cols_idx)
        
        self.num_samples = df["y"].shape[0]
        
        self.data = df
        
        x, y = self.data.loc[:, self.x_cols], \
            self.data.loc[:, "y"]
            
        self.x = x.to_numpy().astype(DTYPE_FLOAT)
        self.y = y.to_numpy().astype(DTYPE_FLOAT)
        self.data = None
    
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        # x, y = self.data.loc[:, self.x_cols].iloc[index], \
        #     self.data.loc[:, "y"].iloc[index]
        # x = x.to_numpy().astype(DTYPE_FLOAT)
        # y = np.array([y], dtype=DTYPE_FLOAT)
        x = self.x[index, :]
        y = self.y[index]
        return x, y
   
class ResprAllSignalsCsvDataset(BaseResprCsvDataset):
    """This dataset is intented to be used when the csv (or dataframe) rows
    contain concatenated [ppg, riav, rifv, riiv] signals. By default
    9600 points (300Hz * 32 s) for ppg and 384 (3 * 4Hz * 32 s) points for
    the induced signal is assumed, and therefore the vector (x) in each row
    will be converted to 4 channels 4 * 9600 (by resampling the induced signals
    @sampling frequency of the ppg). This four channeled sample is returned
    when a sample is requested."""
    def __init__(self, config={}, datasource=None) -> None:
        super().__init__(config, datasource)
        
        
    def prepare_config(self, config):
        self._config =  super().prepare_config(config)
        default_config_items = {
            "ppg_sampling_frequency": 300, #Hz
            "induced_signal_sampling_frequency": 4, #Hz
            "signal_duration": 32, #seconds
            "num_induced_signals": 3 # rifv, riav, riiv
        }
        self._config = fill_missing_values(default_values=default_config_items,
                                           target_container=self._config)
        return self._config
        
    def set_source(self, df):
        # x and y should be set by the nase class
        super().set_source(df)
        self.x = self.check_and_adjust_x(self.x)
        
    def check_and_adjust_x(self, x):
        """Split into component signals and put them in separate channel
        dimensions (after resampling induced signal).

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        ppg_fs = self._config["ppg_sampling_frequency"]
        induced_fs = self._config["induced_signal_sampling_frequency"]
        signal_duration = self._config["signal_duration"]
        num_induced_signals = self._config["num_induced_signals"]
        expected_length = \
            (ppg_fs + num_induced_signals * induced_fs) * signal_duration
        assert x.shape[1] == expected_length
        num_samples = self.x.shape[0]
        
        # resample :
        logger.debug("Resampling")
        ppg_num_points = ppg_fs * signal_duration
        start_idx = ppg_num_points
        step_size = induced_fs * signal_duration
        resampled_induced_signals = []
        for sig_num in range(num_induced_signals):
            current_signal = []
            logger.debug(f"Processing induced signal#{sig_num}")
            for sample_idx in tqdm.tqdm(range(num_samples)):
                
                s = self.x[sample_idx, start_idx:start_idx+step_size]
                s = scipy.signal.resample(x=s, num=ppg_num_points, t=None)
                s = np.expand_dims(s, axis=0)# add channel dim
                s = np.expand_dims(s, axis=0)# add sample num dim
                current_signal.append(s)
            
            
            current_signal = np.concatenate(current_signal, axis=0)
            resampled_induced_signals.append(current_signal)
            
            #move to next signal offset
            start_idx = start_idx + step_size
            
        ppg_signal = np.expand_dims(self.x[:, 0:ppg_num_points], axis=1)
        all_signals = [ppg_signal] + resampled_induced_signals
        all_signals = np.concatenate(all_signals, axis=1) # channel axis
        
        return all_signals
            

        
    
    
class BaseResprDataLoaderComposer:
    def __init__(self, config) -> None:
        self._config = config
        self.dataset = self._config["dataset"] # dataset class name
        self.dataset_path = self._config["dataset_path"]
        self.num_folds = self._config["num_folds"]
        self.val_split = self._config["val_split"]
        self.test_split = self._config["test_split"]
        self.batch_size = self._config["batch_size"]
        self.num_workers = self._config["num_workers"]
        self.random_state = 0
        self.prepare()
    
    def prepare(self):
        raise NotImplementedError()
    
    
    def get_data_loaders(self, current_fold=-1):
        raise NotImplementedError()
        
    def compute_split_sizes(self, n):
        """ Number of subjects alloted to each of the split"""
        
        assert 0 <= self.test_split < 0.9
        assert 0 <= self.val_split < 0.9
        assert (self.val_split + self.test_split) < 0.9
        num_train_ids = int(math.ceil(n * (1-self.test_split-self.val_split)))
        num_val_ids = int(round(n * self.val_split))
        num_test_ids = n - num_train_ids - num_val_ids
        
        assert num_val_ids > 0 and num_test_ids > 0
        
        
        return num_train_ids, num_val_ids, num_test_ids
    
    def split_list(self, input_list, num_fold):
        """Partition `input_list` into three.
        """
        num_train_ids, num_val_ids, num_test_ids = self.compute_split_sizes(
            len(input_list))
        l =  copy.deepcopy(input_list)
        num_max_folds = int(len(input_list)/num_test_ids)
        
        shuffle_times = int(num_fold/num_max_folds)
        rotate_times = (num_fold % num_max_folds)*num_test_ids

        r = random.Random(self.random_state)
        
        for i in range(shuffle_times):
            r.shuffle(l)
        
        l = collections.deque(l)
        l.rotate(rotate_times)
        l = list(l)
        
        train_ids = l[0:num_train_ids]
        val_ids = l[num_train_ids:num_train_ids+num_val_ids]
        test_ids = l[num_train_ids+num_val_ids:]
        
        return train_ids, val_ids, test_ids
        
class ResprCsvDataLoaderComposer(BaseResprDataLoaderComposer):
    
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def _fill_missing_config_values(self):
        # For capnobase mean=-0.9175685194465276 , std=3.883640349389501
        defaults = {
            "x_length": 9600,
            "normalize_x": True,
            "normalization_stats" : {
                "x" : {
                    "mean": -0.91756,
                    "std": 3.88364
                }
            },
            "normalize_mode": "global" # `global` for normalizing each window
            # using the mean and std of the whole dataset. `local` for 
            # normalizing using mean and std of the window
        }
        
        for k, v in defaults.items():
            if k not in self._config:
                logger.warning(f"Key `{k}` was not provided. Using default"
                               f" value: {v}")
                self._config[k] = v
        
        return self._config
        
    
    def prepare(self):
        self._config = self._fill_missing_config_values()
        self.data = self.read_data()
        self.validate_data_structure(self.data)
        self.inspect_data(self.data) # to print stats etc.
        
        if self._config["normalize_x"]:
            logger.info("Normalizing x")
            self.data = self.normalize_x(self.data)
        
        # num unique subjects
        self.subject_ids = list(self.data["subject_ids"].unique())
        self.num_train_ids, self.num_val_ids, self.num_test_ids = \
            self.compute_split_sizes(n = len(self.subject_ids))
            
        assert self.num_folds <= len(self.subject_ids)/self.num_test_ids

    def read_data(self):
        file_path = self._config["dataset_path"]
        if isinstance(file_path, (Path, str)):
            return pd.read_csv(file_path)
        elif isinstance(file_path, (list, tuple)):
            return self.read_multiple(file_list=file_path)
        else:
            raise ValueError()
    
    def read_multiple(self, file_list):
        logger.debug(f"Will be readind data from multiple files. If you are"
                     f"using different datasets, make sure that there is no"
                     f" subject id conflict(overlap), because subject ids are"
                     f" used for splitting data for cross validation.")
        
        raise NotImplementedError()
            
    def validate_data_structure(self, data):
        # also validate x columns start at index #1 (assumin 0 based index) 
        # and continue from there for `x_length`
        df_cols_as_is = [c for c in data.keys()]
        x_len = self._config["x_length"]
        
        for i in range(1, 1+x_len):
            assert df_cols_as_is[i].startswith("x_"), f"Expected column at "
            f"index {i} to start with `x_`"
        
        
        x_cols = sorted([c for c in self.data.keys() if c.startswith("x_")])
        if len(x_cols) != self._config["x_length"]:
            logger.error(f"Number of x columns : {len(x_cols)} is not same"
                         f" as provided in config `x_length`")
        
    
    def normalize_x(self, data):
        if self._config["normalize_mode"] == "global":
            pass # continue to use global stats
        elif self._config["normalize_mode"] == "local":
            return self.normalize_x_locally(data)
        else:
            raise ValueError(f"Unknown normalize_mode: "
                             f"{self._config['normalize_mode']}")
        x_stats = self._config["normalization_stats"]["x"]
        mu = x_stats["mean"]
        std = x_stats["std"]
        x_length = self._config["x_length"]
        if mu is None or std is None:
            logger.info(f"Computing mean / std from data. Provided"
                        f" mean={mu}, std={std}")
            x_temp = data.iloc[:, 1:x_length+1]
            mu = x_temp.mean(axis=0).mean(axis=0)
            var = ((x_temp - mu)**2).mean(axis=0).mean(axis=0)
            std = np.sqrt(var)
        
        logger.info(f"Using mean={mu} , std={std} ")
        
        data.iloc[:, 1:x_length+1] = \
            (data.iloc[:, 1:x_length+1] - mu) / (1e-8 + std)
        
        return data
    
    def normalize_x_locally(self, data):
        x_length = self._config["x_length"]
        x_temp = data.iloc[:, 1:x_length+1].to_numpy()
        mu = x_temp.mean(axis=1, keepdims=True)
        std = x_temp.std(axis=1, keepdims=True)
        
        x_temp = (x_temp - mu)/(1e-8 + std)
        
        data.iloc[:, 1:x_length+1] = x_temp
        
        return data
        
                
    def get_data_loaders(self, current_fold=-1):
        train_loader, val_loader, test_loader = self._get_data_loaders(
            current_fold)
        
        return train_loader, val_loader, test_loader

    def _get_data_loaders(self, current_fold):
        if current_fold == -1:
            raise NotImplementedError()
        subject_ids = collections.deque(self.subject_ids)
        
        subject_ids.rotate(self.num_test_ids*current_fold)
        
        subject_ids = list(subject_ids)
        
        val_offset = self.num_train_ids
        val_end = self.num_train_ids + self.num_val_ids
        
        
        train_ids, val_ids, test_ids = subject_ids[0:self.num_train_ids],\
            subject_ids[val_offset:val_end], subject_ids[val_end:] 
            
        logger.info(f"Subjects -> Train: {train_ids} / Val: {val_ids}"
                    f"/ Test: {test_ids}")
        
        train_loader = self.create_loader(self.data, train_ids, shuffle=True)
        val_loader = self.create_loader(self.data, val_ids, shuffle=False,
                            loader_type="val")
        test_loader = self.create_loader(self.data, test_ids, shuffle=False,
                            loader_type="test")
                            
        return train_loader,val_loader,test_loader
        
    def create_loader(self,  data, subject_ids_subset, shuffle=True,
                      loader_type=None):
        df = data.loc[data["subject_ids"].isin(subject_ids_subset)]
        #>>> TODO: remove assert isinstance(self.dataset, str) 
        dataset = self.get_dataset_instance(df, dataset_type=loader_type)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                shuffle=shuffle, num_workers=self.num_workers)
        return dataloader

    def get_dataset_instance(self, df, dataset_type=None):
        main_dataset_config = self.dataset
        if dataset_type is not None:
            conf_key = f"{dataset_type}_dataset"
            if conf_key in self._config:
                main_dataset_config = self._config[conf_key]
                logger.debug(f"For `{dataset_type}` dataset using "
                            f"this config  {main_dataset_config}.")
            else:
                #>>> logger.error(f"Expected key: `{conf_key}` in config.")
                logger.info(f"For `{dataset_type}` dataset, continuing "
                            f"with main dataset config:"
                               f"{main_dataset_config}")
        if isinstance(main_dataset_config, dict):
            dataset_conf = copy.deepcopy(main_dataset_config)
            # => config has been passed insted of just class name
            class_name = dataset_conf["name"]
            dataset_class = REGISTERED_DATASET_CLASSES[class_name]
            args = dataset_conf["args"]
            kwargs = dataset_conf["kwargs"]
            assert "datasource" not in kwargs
            kwargs["datasource"] = df
        else:
            # `main_dataset_config` is str
            assert isinstance(main_dataset_config, str)
            dataset_class = REGISTERED_DATASET_CLASSES[main_dataset_config]
            args = []
            kwargs = {"datasource": df}
            
        dataset = dataset_class(*args, **kwargs)
        return dataset
    
    def inspect_data(self, data):
        
        bin_step = 2 # breaths/min
        max_bpm = 120
        
        assert isinstance(bin_step, int)
        assert isinstance(max_bpm, int)
        assert max_bpm % bin_step == 0, "bin step must divide max_bpm"
        
        assert data["y"].max() <= max_bpm, f" breaths/min \
            should be <= {max_bpm}"
            
        bins = range(0, max_bpm + 1, bin_step)
        labels = list(bins)[:-1] # i.e. label is start value of each of the bin
        
        y_bins = pd.cut(data["y"], bins=bins, right=False, labels=labels)
        
        y_binned_counts = y_bins.value_counts()
        
        
        counts = []
        for i in range(len(labels)):
            counts.append(y_binned_counts[labels[i]])
        
        max_count = max(counts)
        min_count = max_count# min count (except zero)
        for c in counts:
            if c > 0 and c < min_count:
                min_count = c
        counts = [c if c > 0 else min_count for c in counts]
        
        
        logger.info(f"Bins and counts: {y_binned_counts} / counts: [{counts}]"
                    f" labels: {labels}")
        
        # compute class weights using inverse freq.
        # this is being computed here just for logging. 
        weights = [max_count/c for c in counts]
        logger.info(f"If you were to use inverse frequency class weights. "
                    f" The weights would be: {weights}, for labels: {labels}")
        
        
                
        

class ResprIndexedDataset(Dataset):
    def __init__(self, index_info, data_container) -> None:
        super().__init__()
        self._index_info = index_info
        # this object will be shared (read only)
        self._container: ResprStandardIndexedDataContainer = data_container 
        self.x_vector_length = self._container.indexed_data[
            "_metadata"]["vector_length"]
        
    def __getitem__(self, index) :
        dataset_idx, sample_id, offset = self._index_info[index]
        dataset_id = self._container.indexed_data['dataset_index_to_id'][dataset_idx]
        sample = self._container.indexed_data["datasets"][dataset_id]["samples"][sample_id]
        
        x = sample["x"][offset:offset+self.x_vector_length]
        y = sample["y"][offset]
        
        return x, y
    
    def __len__(self) -> int:
        return len(self._index_info)
class ResprDataLoaderComposer(BaseResprDataLoaderComposer):
    def __init__(self, config) -> None:
        super().__init__(config)
        
    def prepare(self):
        if isinstance(self.dataset_path, (list, tuple)):
            raise NotImplementedError() # multiple datasets support

        container = ResprStandardIndexedDataContainer(config={
            "dataset_file_path": self.dataset_path
        })
        
        self.container = container
    
    
    def get_data_loaders(self, current_fold=-1):
        if current_fold == -1:
            raise NotImplementedError()
        train_split, val_split, test_split = self._partition_dataset_and_samples(current_fold)
        
        logger.info(f"Subjects -> Train: {train_split} / Val: {val_split}"
                    f"/ Test: {test_split}")
        
        train_loader = self.create_loader(train_split, shuffle=True)
        val_loader = self.create_loader(val_split, shuffle=False)
        test_loader = self.create_loader(test_split, shuffle=False)
        
        
        
        return train_loader, val_loader, test_loader
        
    
    
    def _partition_dataset_and_samples(self, current_fold):
        d = self.container.indexed_data
        train_split = defaultdict(lambda  : None )
        val_split =  defaultdict(lambda  : None )
        test_split =  defaultdict(lambda  : None )
        for dataset_id in d["dataset_id_to_index"]:
            dataset = d["datasets"][dataset_id]
            dataset_index = d["dataset_id_to_index"][dataset_id]
            sample_ids = dataset["sample_ids"]
            
            train_ids, val_ids, test_ids = \
                self.split_list(sample_ids, num_fold=current_fold)
                
            train_split[dataset_index] = set(train_ids)
            val_split[dataset_index] = set(val_ids)
            test_split[dataset_index] = set(test_ids)
            
        return train_split, val_split, test_split
                
            
            
        
    def create_loader(self, split_info, shuffle):
        original_index_map = self.container.indexed_data["index"]
        new_idx_map = self.select_by_dataset_and_sample_ids(
            original_index_map, split_info)
        # TODO: may take dataset class from config
        dataset = ResprIndexedDataset(index_info=new_idx_map,
                                      data_container=self.container)
        
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                shuffle=shuffle, num_workers=self.num_workers)
        
        return dataloader
        
    
    def select_by_dataset_and_sample_ids(self, index_map, split_info):
        """ `index_map` is a list of `(<dataset_id>, <sample_id>, <offset>)`
        `split_info` is a map {<dataset_id>:  set of <sample_ids> ...}
        Returns: selected values from index_map
        
        """
        new_idx_map = []
   
        for item in index_map:
            current_dataset_idx, current_sample_id, _ = item
            
            if (current_dataset_idx in split_info) and \
                (current_sample_id in split_info[current_dataset_idx]):
                new_idx_map.append(item)
            
            
        return new_idx_map
        
        
        
        
    
class DatasetAndAugmentationWrapper(Dataset):
    
    def __init__(self, config, datasource=None) -> None:
        self._config = config
        defaults = {
            "underlying_dataset": {
                    "name": None,
                    "args": None,
                    "kwargs": None
                },
            "data_augmentation": None
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        
        dataset_init_schema = self._config["underlying_dataset"]
        dataset_init_schema["kwargs"]["datasource"] = datasource
        self.dataset = DATASET_FACTORY.get(dataset_init_schema)
        
        self.init_augmentation()

    def init_augmentation(self):
        augment_schema = self._config["data_augmentation"]
        self.augmentation = DATA_AUG_FACTORY.get(augment_schema)
    
    def __len__(self) -> int:
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        x, y =  self.dataset.__getitem__(index)
        x_1, x_2, y = self.augmentation(x, y)
        return x_1, x_2, y

class BaseResprCsvDatasetDuplicateX(BaseResprCsvDataset):
    """For using in SimCLR pipeline. But for validation and test dataloders.
    NOTE: both  x (inputs) returned are not augmented."""
    def __init__(self, config={}, datasource=None) -> None:
        super().__init__(config, datasource)
    
    def __getitem__(self, index: int):
        x, y =  super().__getitem__(index)
        x2 = np.copy(x)
        return x, x2, y
        
    def __len__(self) -> int:
        return super().__len__()

REGISTERED_DATASET_CLASSES = {
    "BaseResprCsvDataset": BaseResprCsvDataset,
    "ResprAllSignalsCsvDataset": ResprAllSignalsCsvDataset,
    "DatasetAndAugmentationWrapper": DatasetAndAugmentationWrapper,
    "BaseResprCsvDatasetDuplicateX": BaseResprCsvDatasetDuplicateX
}

DATASET_FACTORY = BaseFactory(
    config={"resource_map": REGISTERED_DATASET_CLASSES}) 
        

if __name__ == "__main__":
    
    def test_csv_dataset(create_train_val_test_split, BaseResprCsvDataset):
        ds = BaseResprCsvDataset({"dataset_path": \
            "../artifacts/2022-11-16_161146/dataset.csv"})
        
        dl = DataLoader(dataset=ds, batch_size=3)
        
        for batch in dl:
            b = batch
            print(type(b))
        sample_ids = [i for i in range(10)]
        train_samples, val_samples, test_samples = \
            create_train_val_test_split(sample_ids, 0.2, 0.2)
        print("Done")
    
    # test_csv_dataset(create_train_val_test_split, BaseResprCsvDataset)
    
    def test_container():
        c1 = ResprStandardIndexedDataContainer(config={
            "dataset_file_path": "../../artifacts/__test/capnobase-win32-stride1-resp-mini-01.pkl"
        })
        c2 = ResprStandardIndexedDataContainer(config={
            "dataset_file_path": "../../artifacts/__test/capnobase-win32-stride1-resp-mini-02.pkl"
        })
    
        c = c1 + c2
        
    test_container()