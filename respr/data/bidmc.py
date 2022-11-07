import os
import re
from pathlib import Path

import heartpy as hp
import pandas as pd
import seaborn as sns
from loguru import logger
from respr.data import StandardDataRecord, BaseDataAdapter
import numpy as np


# BIDMC Dataset : https://physionet.org/content/bidmc/1.0.0/
# TODO: move all paths to external config
ROOT_LOC = Path(os.path.abspath('')).parents[1]
logger.debug(f"ROOT_LOC: {ROOT_LOC}")
DATASET_DIR = ROOT_LOC / "Datasets"
BIDMC_DATSET_CSV_DIR = DATASET_DIR / "bidmc-ppg-and-respiration-dataset-1.0.0"\
                        / "bidmc_csv"


class BidmcDataAdapter(BaseDataAdapter):
    
    def __init__(self, config):
        super().__init__(config)
        # do not change the order
        self.file_prefix = "bidmc_"
        self.file_suffixes = ["_Breaths.csv", "_Numerics.csv",
                              "_Signals.csv", "_Fix.txt"]
        
        
    
    def inspect(self):
        """Checks available files
        
        Args:
            data_root_dir: Directory containing all the csv files
        """
        datadir = Path(self.data_root_dir)
        files = datadir.glob("bidmc_*_*.*")
        file_names = sorted([f.name for f in files])
        
        return file_names
    
    def extract_subject_ids_from_file_names(self, file_names:list):
        ids = set()
        pattern = re.compile("bidmc_[0-9]*_Breaths.csv")
        warnings = []
        for name in file_names:
            if pattern.fullmatch(name):
                id_ = name.split("_")[1]
                if id_ in ids:
                    logger.warning("Id# {id_} repeated in the list")
                else:
                    ids.add(id_)
        return sorted(list(ids))
    
    
    def get_file_paths(self, id_):
        """Returns path of the following files in order.
            'bidmc_<id_>_Breaths.csv',
            'bidmc_<id_>_Numerics.csv',
            'bidmc_<id_>_Signals.csv',
            'bidmc_<id_>_Fix.txt'
        """
        root = Path(self.data_root_dir)
        file_paths = (root / f"{self.file_prefix}{id_}{suffix}" 
                      for suffix in self.file_suffixes)
        return tuple(file_paths)
    
    
    def _load_data(self, id_):
        """Returns contents of the following files in order.
            `csv` files as `pandas.DataFrame` and `txt` file as `string`.
            'bidmc_<id_>_Breaths.csv',
            'bidmc_<id_>_Numerics.csv',
            'bidmc_<id_>_Signals.csv',
            'bidmc_<id_>_Fix.txt'
        
        """
        paths = self.get_file_paths(id_)
        data = []
        for i in range(3):
            logger.debug(f"Loading: {paths[i]}")
            content = pd.read_csv(paths[i])
            data.append(content)

        logger.debug(f"Loading: {paths[3]}")
        with open(paths[3], "r", encoding="utf-8") as f:
            content = f.read()
            data.append(content)
            
        return data
    
    def get(self, id_):
        data = self._load_data(id_)
        data = self.to_standard_format(data, id_)
        return data
    
    def _parse_metadata(self, meta_str):
        keys = ["Signals sampling frequency:",
                "Numerics sampling frequency:"]
        
        
        data = [None] * len(keys)
        
        k_idx = 0
        for line in meta_str.split("\n"):
            if line.startswith(keys[k_idx]):
                v = line.split(":")[1]
                v = v.strip().split()[0]
                v = float(v)
                data[k_idx] = v
                k_idx += 1
            if k_idx == len(data):
                break
        
        # data[0] is signal sampling freq
        # data[1] is numerics sampling freq (respiratory rate signal sampling
        # freq)
        return data
    
    def to_standard_format(self, data, id_):
        signals = data[2]
        numerics = data[1]
        t_numerics = numerics["Time [s]"].to_numpy(dtype=self.signal_dtype)
        t_signals = signals["Time [s]"].to_numpy(dtype=self.signal_dtype)
        ppg = signals[" PLETH"].to_numpy(dtype=self.signal_dtype)
        gt_resp = numerics[" RESP"].to_numpy(dtype=self.signal_dtype)
        meta = data[3]
        signal_fs, numerics_fs = self._parse_metadata(meta)
        
        
        data_std = {
            "id": id_,
            "signals": {
                "ppg" : ppg,
                "gt_resp": gt_resp
            },
            "time":{
                "ppg": t_signals,
                "gt_resp": t_numerics
            },
            "_metadata": {
                "signals":{
                    "ppg" : {
                        "fs" : signal_fs,
                        "has_timestamps": True,
                        "t_loc" : "time/ppg",
                        "t_is_uniform": True, # indicates if signal is 
                        # uniformly sampled everywhere
                        "t_includes_start": True # if the sample includes 
                        # signal value st t=0
                    },
                    "gt_resp": {
                        "fs" : numerics_fs,
                        "t_is_uniform": True,
                        "has_timestamps": True,
                        "t_loc": "time/gt_resp",
                        "t_includes_start": True
                    }
                }
            }
        }
        
        return StandardDataRecord(data_std)

    
if __name__ == "__main__":
    bidmc_data_adapter = BidmcDataAdapter({"data_root_dir": BIDMC_DATSET_CSV_DIR})
    file_names = bidmc_data_adapter.inspect()
    bidmc_data_adapter.extract_subject_ids_from_file_names(file_names)
    bidmc_data_adapter.get_file_paths("01")
    data = bidmc_data_adapter.load_data("01")