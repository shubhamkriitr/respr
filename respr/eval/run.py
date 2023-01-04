
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
    
    
    
if __name__ == "__main__":
    from ._results_file_paths import experiment_num_to_result
    