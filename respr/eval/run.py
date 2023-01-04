
import pickle
import json
import yaml
import pandas as pd
import heartpy
from pathlib import Path
import os
from loguru import logger
import re
import heartpy as hp
from scipy import signal
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap

from respr.pipeline.base import BasePipeline
from respr.util.common import logger
from respr.eval.evaluation import (TYPE_PNN,
    TYPE_SIGNAL_PROCESSING, TYPE_SIGNAL_PROCESSING_2, 
    TYPE_SIGNAL_PROCESSING_2B)
from respr.eval.result_loader import (ResultLoaderPnn, ResultLoaderSignalProc,
        ResultLoaderSignalProcOld, ResultLoaderSignalProcType2B)
from respr.eval.evaluation import BaseResprEvaluator, EvalHelper

DEFAULT_LOADER_MAPPING = {TYPE_PNN: ResultLoaderPnn(),
                          TYPE_SIGNAL_PROCESSING_2: ResultLoaderSignalProc(),
                          TYPE_SIGNAL_PROCESSING: ResultLoaderSignalProcOld(),
                          TYPE_SIGNAL_PROCESSING_2B: ResultLoaderSignalProcType2B()}

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

def convert_loocv_fold_wise_metric_to_df(loocv_fold_wise_metric, fold_numbers):

    df = [np.expand_dims(np.array(fold_numbers), 1)]
    df_cols = ["fold"]
    tags = sorted(list(loocv_fold_wise_metric.keys()))
    for tag in tags:
        d = loocv_fold_wise_metric[tag]
        folds = set([int(f) for f in d])
        mae_list = []
        rmse_list = []
        mean_std_list = []
        for fold in fold_numbers:
            if fold not in folds:
                rmse = np.nan
                mae = np.nan
                mean_std = np.nan
            else:
                fold_str = str(fold).zfill(4)
                rmse = d[fold_str]['rmse']['[metric_rmse]gt:rr_est_pnn']
                mae = d[fold_str]["mae"]["[metric_mae]gt:rr_est_pnn"]
                mean_std = d[fold_str]["mean_std"]
            
            mae_list.append(mae)
            rmse_list.append(rmse)
            mean_std_list.append(mean_std)
        
        mae_arr = np.expand_dims(np.array(mae_list), 1)
            
        rmse_arr = np.expand_dims(np.array(rmse_list), 1)
        mean_std_arr = np.expand_dims(np.array(mean_std_list), 1)
        
        df.append(mae_arr)
        df_cols.append(f"{tag}_mae")
        df.append(rmse_arr)
        df_cols.append(f"{tag}_rmse")
        df.append(mean_std_arr)
        df_cols.append(f"{tag}_mean_std")
    
    df = np.concatenate(df, axis=1)
    df = pd.DataFrame(df, columns=df_cols)
    
    return df

class EvalPipeline(BasePipeline):
    def __init__(self, config={}) -> None:
        super().__init__(config)
        
        self.selected_experiments = self._config["selected_experiments"]
    
    def run(self, *args, **kwargs):
        for tag, experiments_set in self.selected_experiments:
            output_dir = self.output_dir / f"{tag}"
            os.makedirs(output_dir, exist_ok=False)
            try:
                self.run_one_set(experiments_chosen=experiments_set,
                    output_dir=output_dir, tag=tag)
            except Exception as exc:
                logger.exception(exc)
    
    def get_objs(self):
        ev = BaseResprEvaluator()
        eval_helper = EvalHelper()

        return ev, eval_helper
    def run_one_set(self, experiments_chosen, output_dir, tag):
        results_source = []
        ev, eval_helper = self.get_objs()
        
        
        for exp in experiments_chosen:
            if exp in experiment_num_to_result:
                results_source.append(experiment_num_to_result[exp])
            else:
                logger.warning(f"Could not find experiment: {exp}")
        
        all_model_results, loocv_fold_wise_metric \
            = gather_results_from_source(results_source=results_source,
                                         loaders=DEFAULT_LOADER_MAPPING)
        
        # plot MAE 
        std_cutoffs = np.arange(0.5, 50.1, 0.1)
        x_ticks = [i for i in range(0, 50, 2)]
        img_idx = 0
        fig_and_axs = ev.plot_all(all_model_results, metrics=["mae"],
                    figsize=(16, 6), std_cutoffs=std_cutoffs, x_ticks=x_ticks)
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag="mae"
                      )
        img_idx +=1
        
        #plot RMSE
        fig_and_axs = ev.plot_all(all_model_results, metrics=["rmse"],
                                  figsize=(16, 6))
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag="rmse"
                      )
        img_idx +=1
        
        
        # HISTOGRAMS
        model_results_for_histogram = all_model_results
        model_names_used_for_histogram = [m[2] for m in model_results_for_histogram]
        results_for_hist_by_mae, df_for_hist_by_mae = eval_helper.create_histogram(
            model_results_for_histogram, bins={"start":0, "end": 100, "step": 2}, binning_col="$MAE")
        results_for_hist_by_rr, df_for_hist_by_rr = eval_helper.create_histogram(
            model_results_for_histogram, bins={"start":0, "end": 100, "step": 2}, binning_col="rr_est_pnn")
        results_for_hist_by_ground_truth_rr, df_for_hist_by_ground_truth_rr = eval_helper.create_histogram(
            model_results_for_histogram, bins={"start":0, "end": 100, "step": 2}, binning_col="gt")
        results_for_hist_by_confidence, df_for_hist_by_confidence = eval_helper.create_histogram(
            model_results_for_histogram, bins={"start":0, "end": 100, "step": 2}, binning_col="std_rr_est_pnn")
        print(f"Used this models for histogram data preparation: {model_names_used_for_histogram}")
        hist_title = " *".join(model_names_used_for_histogram)
        hist_title = "\n".join(textwrap.wrap(hist_title, 120))
        
        # 
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_mae, x_col="bins", y_cols=["cumulative_percent_samples"],
                           x_label="MAE bins", y_label="% of windows (cumulative)", title=hist_title, title_fontsize=7)
        
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        # FIG
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_mae, x_col="bins", y_cols=["percent_samples"],
                           x_label="MAE bins", y_label="% of windows", title=hist_title, title_fontsize=7)
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        
        # FIG
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_mae, x_col="bins", y_cols=["RR[mean]"],
                           x_label="MAE bins", y_label="RR[mean]", title=hist_title, title_fontsize=7)
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        
        # FIG
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_rr, x_col="bins", y_cols=["MAE"],
                           x_label="RR[estimate] bins", y_label="MAE")
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        
        # FIG
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_ground_truth_rr, x_col="bins", y_cols=["MAE"],
                           x_label="RR[groundtruth] bins", y_label="MAE", title=hist_title, title_fontsize=7)
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        
        # FIG
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_ground_truth_rr, x_col="bins", y_cols=["Uncertainty[mean]"],
                           x_label="RR[groundtruth] bins", y_label="$\sigma$ (uncertainty)", title=hist_title, title_fontsize=7)
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        
        # FIG
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_ground_truth_rr, x_col="bins", y_cols=["RR[mean]"],
                           x_label="RR[groundtruth] bins", y_label="mean predicted RR", title=hist_title, title_fontsize=7)
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        
        # FIG
        fig_and_axs = eval_helper.plot_histogram(results=results_for_hist_by_ground_truth_rr, x_col="bins", y_cols=["percent_samples"],
                           x_label="RR[groundtruth] bins", y_label="% windows", title=hist_title, title_fontsize=7)
        self.save_fig(fig_and_axs=fig_and_axs, output_dir=output_dir,
                      index=img_idx, file_tag=""
                      )
        img_idx += 1
        
        
        
        
    def save_fig(self, fig_and_axs, output_dir, file_tag, index,  prefix="fig_"):
        fig = fig_and_axs[0]
        fig_filename = f"{prefix}{str(index).zfill(6)}_{file_tag}.jpg"
        p = Path(output_dir) / fig_filename
        fig.savefig(p)
        logger.debug(f"Saved: {p}")
        
        
        
        
                
        
    
    
if __name__ == "__main__":
    from respr.eval._results_file_paths import experiment_num_to_result
    import yaml
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-c", "--config", default=None, type=str)
    DEFAULT_PIPELINE = "Pipeline2"
    args = ap.parse_args()
    config_path = args.config
    config_data = None
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    eval_pipeline = EvalPipeline(config=config_data)
    eval_pipeline.run()