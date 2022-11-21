import math
import collections
from torch.utils.data import Dataset, DataLoader
from respr.data.base import BaseDataAdapter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from loguru import logger

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
        self._config = config
        
        self.num_samples = None
        if datasource is None:
            self._dataset_path = self._config["dataset_path"]
            self.initialize_source()
        else:
            self.set_source(datasource)
            
    
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
    
class ResprDataLoaderComposer:
    
    def __init__(self, config) -> None:
        self._config = config
        self.dataset = self._config["dataset"] # dataset class name
        self.num_folds = self._config["num_folds"]
        self.val_split = self._config["val_split"]
        self.test_split = self._config["test_split"]
        self.batch_size = self._config["batch_size"]
        self.num_workers = self._config["num_workers"]
        self.prepare()
        
    
    def prepare(self):
        self.data = pd.read_csv(self._config["dataset_path"])
        self.data = self.normalize_x(self.data)
        
        # num unique subjects
        self.subject_ids = list(self.data["subject_ids"].unique())
        self.num_train_ids, self.num_val_ids, self.num_test_ids = \
            self.compute_split_sizes(n = len(self.subject_ids))
            
        assert self.num_folds <= len(self.subject_ids)/self.num_test_ids
    
    
    def normalize_x(self, data):
        data.iloc[:, 1:9601] = \
            (data.iloc[:, 1:9601] - data.iloc[:, 1:9601].mean(axis=0))\
                /(1e-8 + data.iloc[:, 1:9601].std(axis=0))
        
        return data
        
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
        
        
    def get_data_loaders(self, current_fold=-1):
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
        val_loader = self.create_loader(self.data, val_ids, shuffle=False)
        test_loader = self.create_loader(self.data, test_ids, shuffle=False)
        
        return train_loader, val_loader, test_loader
        
    def create_loader(self,  data, subject_ids_subset, shuffle=True):
        df = data.loc[data["subject_ids"].isin(subject_ids_subset)]
        assert isinstance(self.dataset, str)
        dataset_class = REGISTERED_DATASET_CLASSES[self.dataset]
        dataset = dataset_class(datasource=df)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
                                shuffle=shuffle, num_workers=self.num_workers)
        return dataloader
        

REGISTERED_DATASET_CLASSES = {
    "BaseResprCsvDataset": BaseResprCsvDataset
}      
        
        
    


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
            "dataset_file_path": "../../artifacts/__test/capnobase-win32-stride1-resp-mean-offset-0-num-4.pkl"
        })
        c2 = ResprStandardIndexedDataContainer(config={
            "dataset_file_path": "../../artifacts/__test/capnobase-win32-stride1-resp-mean-offset-4-num-4.pkl"
        })
    
        c = c1 + c2
        
    test_container()