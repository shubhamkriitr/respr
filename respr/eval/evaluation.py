import pickle
import json
import yaml
import pandas as pd
import heartpy
from pathlib import Path
import os
from respr.util import logger
import re
import heartpy as hp
from scipy import signal
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

class BaseEvaluation:
    
    def __init__(self, config) -> None:
        self._config = config
        
    def run(self, *args, **kwargs):
        pass
    
class Evaluation(BaseEvaluation):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        
    
    def run(self, results_file_path):
        pass

class BaseResprEvaluator:
    def __init__(self, config={}):
        self._config = config
        self._prediction_prefix = "rr_est_"
        self._std_prefix = "std_"
        
        possible_suffixes = ["riav", "rifv", "pnn", "fused", "riiv"]
        self._prediction_columns = [f"{self._prediction_prefix}{s}" for s in possible_suffixes] + ["rr_fused"]
        logger.debug(f"self._prediction_columns={self._prediction_columns}")
        
    
    def vary_std_cutoff(self, predictions: pd.DataFrame, std_devs, std_dev_colname):
        # compute 1) percentage of windows retained
        # 2) MAE
        # 3) RMSE
        # create new record for every new std
        
        # assume sorted std
        std_devs = sorted(std_devs)
        df = predictions
        # keys atarting with rr_est
        
        prediction_keys = [k for k in df.keys() if k in self._prediction_columns]
        total_num_of_records = df.shape[0]
        results = {}
        logger.info(f"pred: {prediction_keys}")
        for std in std_devs:
            df_new = df[df[std_dev_colname] <= std]
            retained_records = df_new.shape[0]
            mae_values = self.compute_mae(df_new, "gt", prediction_keys)
            rmse_values = self.compute_rmse(df_new, "gt", prediction_keys)
            
            metrics = self.merge_dicts(mae_values, rmse_values)
            metrics["retained_records_percent"] = float(retained_records*100/total_num_of_records)
            metrics["retained_records_number"] = float(retained_records)
            metrics[std_dev_colname+"_cutoff"] = std
            
            self.add_record(results, metrics)# add values from current run
        
        return results, pd.DataFrame(results)
            
    
    def compute_mae(self, df, gt_key, pred_keys):
        metrics = {}
        metric_name = "mae"
        for pred_k in pred_keys:
            k = f"[metric_{metric_name}]"+gt_key + ":" + pred_k
            metrics[k] = (df[gt_key] - df[pred_k]).abs().mean()
        return metrics

    def compute_rmse(self, df, gt_key, pred_keys):
        metrics = {}
        metric_name = "rmse"
        for pred_k in pred_keys:
            k = f"[metric_{metric_name}]"+gt_key + ":" + pred_k
            # metrics[k] = np.sqrt(((df[gt_key] - df[pred_k])**2).mean())
            metrics[k] = ((df[gt_key] - df[pred_k])**2).mean()**.5
        return metrics


    def add_record(self, cont, new_entry):
        for k in new_entry:
            try:
                cont[k].append(new_entry[k])
            except KeyError:
                cont[k] = [new_entry[k]]

        return cont

    def merge_records(self, cont, new_entry):
        for k in new_entry:
            try:
                cont[k] = cont[k] + new_entry[k]
            except KeyError:
                cont[k] = new_entry[k]

        return cont
    
    def merge_dicts(self, dict_1, dict_2):
        for k, v in dict_2.items():
            if k not in dict_1:
                dict_1[k] = v
            else:
                # logger.debug(f"Not merging key : {k} from second dict")
                pass
        return dict_1
    
    
    
    def plot_samples_retained_vs_std_cutoff(self, df):
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8,  6),
                        layout="constrained")
        axs.plot(df["std_rr_fused_cutoff"], df["retained_records_percent"])
        axs.axvline(x=4.0, color="red", linestyle="--")
        # axs.set_title("Retained Windows (%)")
        axs.set_ylabel("Windows retained (%)")
        axs.set_xlabel("Std. dev. cutoff (in breaths/min)")
        axs.set_ylim([0, 105])
        axs.set_yticks([i for i in range(0, 101, 5)])
        axs.set_xticks([i for i in range(0, 35, 2)])
    
    
    def plot_y_vs_x(self, datalist, x_label, y_label, figsize=(8, 6), x_lim=None, y_lim=None, x_ticks=None, y_ticks=None, title=None):
        """
        [
            {
                "data": <a dataframe>,
                "tag": <to be used in legend>
                "y_colname": <column to be used for y values>,
                "x_colnmae": <column to be used for x values>
            }
            ...
        
        ]
        """
        
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=figsize,
                        layout="constrained")
        d = datalist[0]
        
        
        
        if title is not None: axs.set_title(title)
        
        handles = []
        legend_items = []
        
        for d in datalist:
            df = d["data"]
            x_col = d["x_colname"]
            y_col = d["y_colname"]
            tag = d["tag"]
            handle, =  axs.plot(df[x_col], df[y_col])
            handles.append(handle)
            legend_items.append(tag)
        axs.set_ylabel(y_label)
        axs.set_xlabel(x_label)
            
        if x_lim is not None: axs.set_xlim(x_lim)
        if y_lim is not None: axs.set_ylim(y_lim)
        if y_ticks is not None: axs.set_yticks(y_ticks)
        if x_ticks is not None: axs.set_xticks(x_ticks)
        
        axs.legend(handles, legend_items)
        
        return (fig, axs)
        
    
    
        
    
    
    
        