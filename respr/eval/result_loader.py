import pickle
import json
import yaml
import pandas as pd
import heartpy
from pathlib import Path
import os
from respr.util.common import logger
import re
import heartpy as hp
from scipy import signal
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class BaseResultLoader:
    def __init__(self):
        pass
    
    def load(self, path):
        pass

class ResultLoaderSignalProc(BaseResultLoader):
    def __init__(self):
        pass
    
    def load(self, path):
        all_data = {}
        with open(path, "rb") as f:
            res = pickle.load(f)
        for d in res:
            self.merge_records(all_data, d["output"])
        all_data = pd.DataFrame(all_data)
        # all_data.describe()
        return all_data
    
    def merge_records(self, cont, new_entry):
        for k in new_entry:
            try:
                cont[k] = cont[k] + new_entry[k]
            except KeyError:
                cont[k] = new_entry[k]

        return cont
    
class ResultLoaderSignalProcOld(ResultLoaderSignalProc):
    def __init__(self):
        super().__init__()
    
    def load(self, path):
        all_data = super().load(path)
        all_data["std_rr_fused"] = self.get_std_dev(all_data)
        return all_data
    
    # add standard deviation TODO add std dev in the prediction pipeline
    def get_std_dev(self, df):
        r1 = df["rr_est_riav"]
        r2 = df["rr_est_rifv"]
        r3 = df["rr_est_riiv"]
        mu = (r1+r2+r3)/3.0
        std = (((r1-mu)**2 + (r2-mu)**2 + (r3-mu)**2)/3)**.5
        return std
    

class ResultLoaderPnn(BaseResultLoader):
    def __init__(self):
        self.ev = BaseResprEvaluator({})
    
    def selector(self, key: str):
        if not key.startswith("predictions_"):
            return False
        if not "fold" in key:
            return False
        if not key.endswith(".csv"):
            return False
        return True


    def get_file_list(self, root_dir, selector, search_levels=[]):
        if len(search_levels) != 0:
            raise NotImplementedError()
        root = Path(str(root_dir))
        files = os.listdir(root)
        print([(type(f), f) for f in files])

        selected_file_paths = sorted( [root/ f for f in files if selector(f)])

        return selected_file_paths
    
    def load(self, path, selector=None, result_container={}):
        if selector is None:
            selector = self.selector
        dfr = []
        file_list = self.get_file_list(path, selector)
        logger.debug(file_list)
        for fpath in file_list:
            logger.info(f"Loading: {fpath}")
            df_ = pd.read_csv(fpath)
            dfr.append(df_)
            logger.info(f"shape: {df_.shape}")
            mae_value = self.ev.compute_mae(df_, "gt", ["rr_est_pnn"])
            rmse_value = self.ev.compute_rmse(df_, "gt", ["rr_est_pnn"])
            mean_std = df_["std_rr_est_pnn"].mean()
            info_to_add = {
                "mae": mae_value,
                "rmse": rmse_value,
                "mean_std": mean_std,
                "shape": df_.shape
            }
            self.add_data_to_foldwise_metric_container(result_container,
                                                      fpath,
                                                      info_to_add)
            logger.info(mae_value)
            logger.info(rmse_value)
        dfr = pd.concat(dfr)
        logger.info(f"TOTAL records: {dfr.shape}")
        return dfr
    
    def add_data_to_foldwise_metric_container(self, container, predictions_file_path, info_to_add):
        
        # find fold number from file_path
        fold = str(predictions_file_path).split("fold_")[1][0:4] 
        logger.debug(f"FOLD: {fold} : fpath : {predictions_file_path}")
        if fold in container:
            logger.error(f"fold {fold} appeared more than once")
        container[fold] = info_to_add
        
        
