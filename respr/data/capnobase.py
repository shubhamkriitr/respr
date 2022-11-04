
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
from respr.data import StandardDataRecord

# CapnoBase Dataset: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT
ROOT_LOC = Path(os.path.abspath('')).parents[1]
DATASET_DIR = ROOT_LOC / "Datasets"
CAPNOBASE_DATASET_CSV_DIR = DATASET_DIR / "capnobase-dataverse" \
                        / "data" / "csv"

logger.info(f"CAPNOBASE_DATASET_CSV_DIR:  {CAPNOBASE_DATASET_CSV_DIR}")


class CapnobaseDataAdapter:
    
    def __init__(self, config):
        self.config = config
        self.data_root_dir = self.config["data_root_dir"]
        # do not change the order
        self.file_prefix = ""
        self.file_suffixes = ["_8min_labels.csv", "_8min_meta.csv",
                              "_8min_param.csv", "_8min_reference.csv",
                              "_8min_SFresults.csv", "_8min_signal.csv"]
    
    def inspect(self):
        """Checks available files
        
        Args:
            data_root_dir: Directory containing all the csv files
        """
        datadir = Path(self.data_root_dir)
        files = datadir.glob("[0-9]"*4 + "*.csv")
        file_names = sorted([f.name for f in files])
        
        return file_names
    
    def extract_subject_ids_from_file_names(self, file_names:list):
        ids = set()
        pattern = re.compile(r"[0-9]{4}_[0-9]min_signal\.csv")
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
    
    
    def load_data(self, id_):
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
    
    

    
