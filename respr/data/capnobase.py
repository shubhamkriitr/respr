
import pandas as pd
import heartpy
from pathlib import Path
import os
from loguru import logger
import re
import heartpy as hp
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from respr.data.base import StandardDataRecord, BaseDataAdapter
import h5py
import copy

# CapnoBase Dataset: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT
ROOT_LOC = Path(os.path.abspath('')).parents[1]
DATASET_DIR = ROOT_LOC / "Datasets"
CAPNOBASE_DATASET_CSV_DIR = DATASET_DIR / "capnobase-dataverse" \
                        / "data" / "csv"
CAPNOBASE_DATASET_MAT_DIR = DATASET_DIR / "capnobase-dataverse" \
                        / "data" / "mat"

logger.info(f"CAPNOBASE_DATASET_CSV_DIR:  {CAPNOBASE_DATASET_CSV_DIR}")
INT_DTYPE = np.int32

class CapnobaseDataAdapter(BaseDataAdapter):
    
    def __init__(self, config):
        super().__init__(config)
        # do not change the order
        self.file_prefix = ""
        self.file_suffixes = ["_8min_labels.csv", "_8min_meta.csv",
                              "_8min_param.csv", "_8min_reference.csv",
                              "_8min_SFresults.csv", "_8min_signal.csv"]
        self.input_file_regex = r"[0-9]{4}_[0-9]min_signal\.csv"
        self.input_file_glob = "[0-9]"*4 + "*.csv"
        
    
    def inspect(self):
        """Checks available files
        
        Args:
            data_root_dir: Directory containing all the subject files
        """
        datadir = Path(self.data_root_dir)
        files = datadir.glob(self.input_file_glob)
        file_names = sorted([f.name for f in files])
        
        return file_names
    
    def extract_subject_ids_from_file_names(self, file_names:list):
        ids = set()
        pattern = re.compile(self.input_file_regex)
        warnings = []
        for name in file_names:
            if pattern.fullmatch(name):
                id_ = name.split("_")[0]
                if id_ in ids:
                    logger.warning("Id# {id_} repeated in the list")
                else:
                    ids.add(id_)
        return sorted(list(ids))
    
    
    def get_file_paths(self, id_):
        """Returns path of the following files in order.
            <id_>_8min_labels.csv   
            <id_>_8min_meta.csv     
            <id_>_8min_param.csv    
            <id_>_8min_reference.csv
            <id_>_8min_SFresults.csv
            <id_>_8min_signal.csv 
        """
        root = Path(self.data_root_dir)
        file_paths = (root / f"{self.file_prefix}{id_}{suffix}" 
                      for suffix in self.file_suffixes)
        return tuple(file_paths)
    
    
    def _load_data(self, id_):
        """Returns contents of the following files in order.
            <id_>_8min_labels.csv   
            <id_>_8min_meta.csv     
            <id_>_8min_param.csv    
            <id_>_8min_reference.csv
            <id_>_8min_SFresults.csv
            <id_>_8min_signal.csv
        """
        paths = self.get_file_paths(id_)
        data = []
        for i in range(6):
            logger.debug(f"Loading: {paths[i]}")
            content = pd.read_csv(paths[i])
            data.append(content)

        # logger.debug(f"Loading: {paths[3]}")
        # with open(paths[3], "r", encoding="utf-8") as f:
        #     content = f.read()
        #     data.append(content)
            
        return data
    
    

class CapnobaseMatDataAdapter(CapnobaseDataAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.file_prefix = ""
        self.file_suffixes = ["_8min.mat"]
        self.input_file_regex = r"[0-9]{4}_[0-9]min\.mat"
        self.input_file_glob = "[0-9]"*4 + "*.mat"
    

    def get_file_paths(self, id_):
        """Returns path of the following file(s):
            <id_>_8min.mat
            
        """
        root = Path(self.data_root_dir)
        file_paths = (root / f"{self.file_prefix}{id_}{suffix}" 
                      for suffix in self.file_suffixes)
        return tuple(file_paths)
    
    
    def _load_data(self, id_):
        """Returns contents of the following file(s).
            <id_>_8min.mat
        """
        paths = self.get_file_paths(id_)
        mat_file_loc = paths[0]
        with  h5py.File(mat_file_loc, 'r') as f:
            data = self._extract_data_items(f, id_)
        
        return data
            
    def _extract_data_items(self, file_handle, id_):
        f = file_handle
        ppg_peak_x = np.array(
            f['labels']['pleth']['peak']['x'][:, 0], dtype=np.int64)
        ppg = np.array(
            f['signal']['pleth']['y'][0, :], dtype=self.signal_dtype)
        
        signal_fs = f['param']['samplingrate']['pleth'][0, 0] #Hz
        t_duration_whole_signal = 8 * 60 # in secs
        num_data_points = int(t_duration_whole_signal * signal_fs) + 1 
        # +1 for  t = 0
        
        t = np.linspace(0, t_duration_whole_signal, num_data_points)
        
        assert (t.shape == ppg.shape)
        
        # respiratory signals (based on capnometry data)
        rr_timestamp = np.array(f['reference']['rr']['co2']['x'])[:, 0]
        rr_value = np.array(f['reference']['rr']['co2']['y'])[:, 0]
        ppg_artifacts_time_chunks, ppg_has_artifacts = \
            self._extract_artifacts(f, timesteps=t) 
        
        data_std = {
            "id": id_,
            "signals": {
                "ppg" : ppg,
                "gt_resp": rr_value,
            },
            "time":{
                "ppg_peak_time_index": ppg_peak_x, # currently not in use 
                "ppg": t,
                "gt_resp": rr_timestamp,
                #PPG artifacts timings are stored as 
                #[start_1, end_1, start_2, end_2....]
                # start_i and end_i are start and end times of the 
                # artifact chunks
                "ppg_artifacts": ppg_artifacts_time_chunks
            },
            "_metadata": {
                "signals":{
                    "ppg" : {
                        "fs" : signal_fs,
                        "has_timestamps": True,
                        "t_loc" : "time/ppg",
                        "t_is_uniform": True, # indicates if signal is 
                        # uniformly sampled everywhere
                        "t_includes_start": True, # if the sample includes 
                        "has_artifacts": ppg_has_artifacts,
                        "artifacts_loc": "time/ppg_artifacts"
                        # signal value st t=0
                    },
                    "gt_resp": {
                        "fs" : None,
                        "t_is_uniform": False,
                        "has_timestamps": True,
                        "t_loc": "time/gt_resp",
                        "t_includes_start": True
                    }
                }
            }
        }
        
        return StandardDataRecord(data_std)
    
    
    def _extract_artifacts(self, file_handle, timesteps):
        ppg_artifacts_time_indices =\
            file_handle['labels']['pleth']['artif']['x'][:]
        ppg_artifacts_time_indices =  ppg_artifacts_time_indices.astype(
            INT_DTYPE
        )
        
        artifacts_timestep_chunk = None
        has_artifacts = False
        
        # this is for no artifacts case
        if len(ppg_artifacts_time_indices.shape) == 1:
            if not np.all( ppg_artifacts_time_indices == 0):
                logger.warning(f"Unexpected format")
            if ppg_artifacts_time_indices.shape[0] != 2:
                logger.warning(f"Unexpected shape")
            return artifacts_timestep_chunk, has_artifacts
            
        ppg_artifacts_time_indices = np.ravel(ppg_artifacts_time_indices)
        
        # make index start from zero (in the raw dataset index is assumed to 
        # start from 1)
        ppg_artifacts_time_indices = ppg_artifacts_time_indices - 1
        has_artifacts = True
        artifacts_timestep_chunk = timesteps[ppg_artifacts_time_indices]
        
        return artifacts_timestep_chunk, has_artifacts
            
        
    def get(self, id_):
        return self._load_data(id_)
    
    
    def print_keys(self, current_item, name, level, key_history):
        prefix = level * "--" + "> "
        suffix = "\n"
        key_history = copy.deepcopy(key_history)
        if level > 0:
            key_history.append(f"['{name}']")
        else:
            key_history.append(f"{name}")
        key_history_str = "".join(key_history)
        
        print(f"{prefix}{key_history_str}{suffix}")
        
        keys = None
        try:
            keys = sorted(list(current_item.keys()))
        except (KeyError, AttributeError) as exc:
            return
        if keys is None:
            return
        for k in keys:
            self.print_keys(current_item[k], k, level + 1, key_history)
            
    
    
if __name__ == "__main__":
    da = CapnobaseMatDataAdapter(
        {"data_root_dir": CAPNOBASE_DATASET_MAT_DIR})
    file_names = da.inspect()
    subject_ids = da.extract_subject_ids_from_file_names(file_names)
    print(file_names, len(file_names))
    print(subject_ids)
    idx = 0
    da._load_data(subject_ids[idx])
