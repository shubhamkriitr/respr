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

TYPE_SIGNAL_PROCESSING = 0
TYPE_SIGNAL_PROCESSING_2 = 2
TYPE_PNN = 1

class BaseResprEvaluator:
    def __init__(self, config={}):
        self._config = config
        self._prediction_prefix = "rr_est_"
        self._std_prefix = "std_"
        
        possible_suffixes = ["riav", "rifv", "pnn", "fused", "riiv"]
        self._prediction_columns = [f"{self._prediction_prefix}{s}" for s in possible_suffixes] + ["rr_fused"]
        logger.debug(f"self._prediction_columns={self._prediction_columns}")
        self._type_to_prediction_col = {
        TYPE_PNN: "rr_est_pnn",
        TYPE_SIGNAL_PROCESSING: "rr_fused",
        TYPE_SIGNAL_PROCESSING_2: "rr_est_fused"
        }
        
    
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
            med_ae_values = self.compute_metric("med_ae", df_new, "gt", prediction_keys)
            
            metrics = self.merge_dicts(mae_values, rmse_values)
            metrics = self.merge_dicts(metrics, med_ae_values)
            
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
    
    def compute_metric(self, metric_name, df, gt_key, pred_keys):
        """metric_func has signature metric_func(input, target)"""
        if metric_name == "med_ae":
            metric_func = self.median_abs_err
        else:
            raise NotImplementedError()
        metrics = {}
        for pred_k in pred_keys:
            k = f"[metric_{metric_name}]"+gt_key + ":" + pred_k
            metrics[k] = metric_func(df[gt_key], df[pred_k])
        return metrics
    
    def median_abs_err(self, input, target):
        return (input - target).abs().median()

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
    
    
    def plot_y_vs_x(self, datalist, x_label, y_label, figsize=(8, 6), x_lim=None, y_lim=None, x_ticks=None, y_ticks=None, title=None, fig_axs=None):
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
        if fig_axs is None:
            fig, axs = plt.subplots(ncols=1, nrows=1, figsize=figsize,
                            layout="constrained")
        else:
            fig, axs = fig_axs
        d = datalist[0]
        
        
        
        if title is not None: axs.set_title(title)
        
        handles = []
        legend_items = []
        
        for d in datalist:
            df = d["data"]
            x_col = d["x_colname"]
            y_col = d["y_colname"]
            tag = d["tag"]
            if isinstance(y_col, (list, tuple)):
                assert isinstance(tag, (list, tuple))
                assert len(tag) == len(y_col)
            else:
                y_col = [y_col]
                tag = [tag]
            for i in range(len(y_col)):
                y_col_i = y_col[i]
                tag_i = tag[i]
                handle, =  axs.plot(df[x_col], df[y_col_i])
                handles.append(handle)
                legend_items.append(tag_i)
        axs.set_ylabel(y_label)
        axs.set_xlabel(x_label)
            
        if x_lim is not None: axs.set_xlim(x_lim)
        if y_lim is not None: axs.set_ylim(y_lim)
        if y_ticks is not None: axs.set_yticks(y_ticks)
        if x_ticks is not None: axs.set_xticks(x_ticks)
        
        axs.legend(handles, legend_items)
        
        return (fig, axs)
    
    def plot_all(self, results, metrics=["mae"], std_cutoffs=None, x_ticks=None, figsize=(16, 6)):
    
        # PLotting %retained
        datalist = []
        std_cutoffs = np.arange(0.5, 30.1, 0.1) if std_cutoffs is None else std_cutoffs
        x_ticks = [i for i in range(0, 35, 2)] if x_ticks is None else x_ticks
        
        for df, type_code, tag in results:
            std_dev_colname = "std_"+ self._type_to_prediction_col[type_code]
            logger.debug(f"std_dev_colname={std_dev_colname}")
            _, df_processed = self.vary_std_cutoff(df, std_cutoffs, std_dev_colname)
            
            y_col = "retained_records_percent"
            x_col = std_dev_colname + "_cutoff"
            datalist_entry = {
                "data": df_processed,
                "x_colname": x_col,
                "y_colname": y_col,
                "tag": tag
            }
            datalist.append(datalist_entry)
        
        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize,
                            layout="constrained")
        self.plot_y_vs_x(
        datalist=datalist,
        x_label="$\sigma$ threshold  ($breaths/min$)",
        y_label="windows retained (%)",
        x_ticks=x_ticks,
        y_ticks=[i for i in range(0, 101, 5)],
        fig_axs=(fig, axs[0])
        )
        
        axs[0].axvline(x=4.0, color="red", linestyle="--")
        axs[0].grid(color = 'green', linestyle = '--', linewidth = 0.3)
        
        if len(metrics) > 1:
            raise NotImplementedError()
            
        metric_to_plot = metrics[0]
        for d  in datalist:
            prediction_col_name = d["x_colname"][4:-7] # e.g std_rr_est_nn_cutoff -> rr_est_pnn
            d["y_colname"] = f"[metric_{metric_to_plot}]gt:{prediction_col_name}"
        
        
        self.plot_y_vs_x(
        datalist=datalist,
        x_label="$\sigma$ threshold  ($breaths/min$)",
        y_label=f"{metric_to_plot}",
        x_ticks=x_ticks,
        fig_axs=(fig, axs[1])
        )
        
        axs[1].axvline(x=4.0, color="red", linestyle="--")
        axs[1].grid(color = 'green', linestyle = '--', linewidth = 0.3)
        