import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from respr.util import logger
from respr.eval.result_loader import (ResultLoaderPnn, ResultLoaderSignalProc,
                                      ResultLoaderSignalProcOld)
TYPE_SIGNAL_PROCESSING = 0
TYPE_SIGNAL_PROCESSING_2 = 2
TYPE_PNN = 1


DEFAULT_LOADER_MAPPING = {TYPE_PNN: ResultLoaderPnn(),
                          TYPE_SIGNAL_PROCESSING_2: ResultLoaderSignalProc(),
                          TYPE_SIGNAL_PROCESSING: ResultLoaderSignalProcOld()}

def gather_results_from_source(results_source, loaders=DEFAULT_LOADER_MAPPING):
    logger.info(f"Loader mapping: {loaders}")
    all_model_results = []
    loocv_fold_wise_metric = {}
    for source, type_code, tag in results_source:
        if type_code == TYPE_PNN:
            if tag in loocv_fold_wise_metric:
                logger.warning(f"Duplicate tag: {tag}")
            info_dict = {}
            data = loaders[type_code].load(source, result_container=info_dict)
            loocv_fold_wise_metric[tag] = info_dict
        else:
            data = loaders[type_code].load(source)
        all_model_results.append((data, type_code, tag))
    
    return all_model_results, loocv_fold_wise_metric

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




# TODO: merge with BaseResprEvaluator or extend it
# To create histogram  (with binning MAE  but y -values can be RR/ Uncertainty etc. )
class EvalHelper:
    def get_bin_indices(self, x, bins):
        """`x` is a 1D numpy array. `bins` is a dictionary
        having following structure: 
            {
                "start": <start value>, # inclusive
                "end": <end value>, # exclusive
                "step": <step size>
            }
        returns a dictionry of the form:
        {
            "bins": <list of `list of two values [start and end) of bin`>
            "indices": <list of `arrays (of indices for the corresponding bin)` >
        }
        """
        assert len(x.shape) == 1, f"x must be a 1D array"
        if isinstance(bins, dict):
            bins = self.creat_bins(**bins)
        else:
            raise NotImplementedError()
        
        indices = []
        index_arr = np.arange(x.shape[0])
        for start_value, end_value in bins:
            condition_met = np.logical_and(x >= start_value, x < end_value)
            bin_indices = index_arr[condition_met]
            indices.append(bin_indices)
        
        return {
            "bins": bins,
            "indices": indices
        }
    
    def creat_bins(self, start, end, step):
        s = start
        n = start + step
        bins = []
        while n <= end:
            bins.append([s, n])
            s = n
            n += step
        return bins
    
    def create_histogram(self, all_model_results,  bins, binning_col="$MAE"):
        """
        `all_model_results` is a list of `tuple, 
        (data, type_code, tag)`. where `data` is
        the results data frame containing ground truth (column `gt`), 
        uncertainty, predicted respiratory rate.
        
        """
        
        
        # project and create a single data frame
        all_results_df = []
        for df, type_, tag_ in all_model_results:
            try:
                df = df[["rr_est_pnn", "std_rr_est_pnn", "gt"]] #project
            except KeyError:
                logger.error(f"Skipping : {tag_} due to KeyError")
                continue
            all_results_df.append(df)
        all_results_df = pd.concat(all_results_df)
        
        
        all_results_df["MAE"] = np.abs(all_results_df["rr_est_pnn"] - all_results_df["gt"])
        mae_array = all_results_df["MAE"].to_numpy()
        
        if binning_col.startswith("$"):
            assert binning_col == "$MAE"
            
            x = mae_array
        else:
            logger.debug(f"Using column {binning_col} for binning.")
            x = all_results_df[binning_col].to_numpy()
            
        bins_and_indices = self.get_bin_indices(x, bins)
        bins = bins_and_indices.pop("bins")
        indices = bins_and_indices.pop("indices")
        
        results = {
            "bins": [], # lisst of str
            "MAE": [], # center
            "RR[mean]": [],
            "Uncertainty[mean]": [],
            "num_samples": [],
            "percent_samples": []
        }
        
        rr_array = all_results_df["rr_est_pnn"].to_numpy()
        std_array = all_results_df["std_rr_est_pnn"].to_numpy()
        mae_array = all_results_df["MAE"].to_numpy()
        for idx, b in enumerate(bins):
            start, end = b
            bin_name = f"{start:.2f}-{end:.2f}"
            index_arr = indices[idx]
            if index_arr.shape[0] < 1:
                print(f"Skipping bin: {b}")
                continue
                
            rr = rr_array[index_arr].mean()
            uncertainty = std_array[index_arr].mean()
            bin_mae = mae_array[index_arr].mean()
            
            results["bins"].append(bin_name)
            results["MAE"].append(bin_mae)
            results["RR[mean]"].append(rr)
            results["Uncertainty[mean]"].append(uncertainty)
            num_s = index_arr.shape[0]
            percent_s= (num_s/x.shape[0])*100
            results["num_samples"].append(num_s)
            results["percent_samples"].append(percent_s)
            
        
        return results, all_results_df
    
    def plot_histogram(self, results: dict, x_col, y_cols,
                       x_label, y_label, figsize=(8, 6), x_lim=None,
                       y_lim=None, x_ticks=None, y_ticks=None,
                       title=None, fig_axs=None):
        if fig_axs is None:
            fig, axs = plt.subplots(ncols=1, nrows=1, figsize=figsize,
                            layout="constrained")
        else:
            fig, axs = fig_axs
        
        
        
        if title is not None: axs.set_title(title)
        
        handles = []
        legend_items = []
        if isinstance(y_cols, str):
            y_cols = [y_cols]
        
        
        num_bars_per_bin = len(y_cols)
        bar_width = 0.2
        bar_total_width = bar_width*num_bars_per_bin
        
        x_axis_num = np.arange(len(results[x_col]))
        bar_offset = 0
        
        for y_colname in y_cols:

            tag_i = y_colname
            handle =  axs.bar(x=x_axis_num+bar_offset,
                               height=results[y_colname],
                               width=bar_total_width) #,
                               # label=y_colname
            handles.append(handle)
            legend_items.append(tag_i)
            bar_offset += bar_width
        axs.set_xticks(x_axis_num, results[x_col], rotation=65)
        axs.set_ylabel(y_label)
        axs.set_xlabel(x_label)
            
        if x_lim is not None: axs.set_xlim(x_lim)
        if y_lim is not None: axs.set_ylim(y_lim)
        if y_ticks is not None: axs.set_yticks(y_ticks)
        # if x_ticks is not None: axs.set_xticks(x_ticks)
        
        axs.legend(handles, legend_items)
        
        return (fig, axs)
        