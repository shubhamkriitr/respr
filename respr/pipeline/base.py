import os
from pathlib import Path
import yaml
import json
import respr.util.common as cutl
from respr.util import logger
from respr.util.common import (get_timestamp_str, PROJECT_ROOT)
import pickle
from respr.data import DATA_ADAPTER_FACTORY
from respr.data.bidmc import BidmcDataAdapter, BIDMC_DATSET_CSV_DIR
from respr.core.process import (PpgSignalProcessor, PROCESSOR_FACTORY)
from respr.core.pulse import PulseDetector
import heartpy as hp
import traceback
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from respr.core.ml.models import ML_FACTORY
from torch.utils.data import DataLoader
import copy

DTYPE_FLOAT = np.float32
CONF_FEATURE_RESAMPLING_FREQ = 4 # Hz. (for riav/ rifv/ riiv resampling)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
class BasePipeline:
    
    def __init__(self, config={}) -> None:
        self._config = config
        # TODO: remove the following override
        self.creation_time = get_timestamp_str()
        if "output_dir" not in self._config:
            self._config["output_dir"] = Path("../../artifacts")
        else:
            self._config["output_dir"] = Path(self._config["output_dir"])
        os.makedirs(self._config["output_dir"], exist_ok=True)
        
        self.root_output_dir = self._config["output_dir"]
        self.output_dir = self.root_output_dir / self.creation_time
        os.makedirs(self.output_dir, exist_ok=False)
        self._log_path = self.output_dir / f"{self.creation_time}_logs.log"
        self._log_sink = logger.add(self._log_path)
        self._buffer = {}
        
        
    
    def run(self, *args, **kwargs):
        pass
    
    def close(self):
        #clean up
        self._close_log_sink()
    
    def _close_log_sink(self):
        logger.remove(self._log_sink)
    
class Pipeline(BasePipeline):
    def __init__(self, config={}) -> None:
        super().__init__(config)
        
        
        self._fill_missing_instructions()
        self._instructions = self._config["instructions"]
        
        cutl.save_yaml(self._config, self.output_dir/"config.yaml")
    
    def _fill_missing_instructions(self):
        default_values = {
            "window_duration" : 32, # in seconds
            "window_step_duration" : 1, # window stride in seconds
            "expected_signal_duration" : 480,
            "window_type": "hamming",
            "ground_truth_mode": "mean"
        }
        
        for k, v in default_values.items():
            if k not in self._config["instructions"]:
                self._config["instructions"][k] = v
        
        return self._config
    
    def run(self, *args, **kwargs):
        # bidmc_data_adapter = BidmcDataAdapter(
        #     {"data_root_dir": BIDMC_DATSET_CSV_DIR})
        bidmc_data_adapter = DATA_ADAPTER_FACTORY.get(
            self._config["data_adapter"])
        file_names = bidmc_data_adapter.inspect()
        subject_ids = bidmc_data_adapter.extract_subject_ids_from_file_names(file_names)
        
        
        results = []
        errors = []
        
        n_samples = self._config["instructions"]["num_samples_to_process"]
        if n_samples is not None:
            sample_idx_offset = \
                self._config["instructions"]["sample_index_offset"]
            sample_idx_end = sample_idx_offset + n_samples
            subject_ids = subject_ids[sample_idx_offset:sample_idx_end]

        # make sure all ids are unique
        assert len(set(subject_ids)) == len(subject_ids)
        
        # give access to subject ids and data adapter to subclasses, in case
        # some precomputation are required. e.g. calculation of global statis-
        # tics
        self.on_data_loaded(subject_ids, bidmc_data_adapter)
        
        
        for idx, subject_id in enumerate(subject_ids):
            # if subject_id != "41": FIXME: investigate this
            #     continue
            
            
            logger.info(f"Processing subject#{subject_id}")
            data = bidmc_data_adapter.get(subject_id)
            try:
                output = self.process_one_sample(data)
                results.append({
                    "idx": idx,
                    "sample_id": subject_id,
                    "output": output
                    
                })
            except Exception:
                logger.error(traceback.format_exc())
                errors.append({
                    "idx": idx,
                    "sample_id": subject_id
                })
                continue
        
        if len(errors) > 0:
            logger.error(errors)
            
        
        # add results 
        self._buffer["results"] = results
        self._buffer["errors"] = errors
        
        # save results
        output_file = self.output_dir/ \
            ("output_" + get_timestamp_str() + ".pkl")
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        
        # close the pipeline properly
        self.close()
        
    def on_data_loaded(self, subject_ids, data_adapter):
        pass
        
    def process_one_sample(self, data):
        signal_name = "ppg"
        
        # params
        #125 # Sampling freq. TODO extract from data
        fs = data.value()["_metadata"]["signals"]["ppg"]["fs"]
        fs = int(fs)
        
        ppg = data.value()["signals"]["ppg"]
        gt_resp_full = data.get("signals/gt_resp")
        
        # sampling freq of respiratory rate (in Hz.)
        resp_fs =  data.value()["_metadata"]["signals"]["gt_resp"]["fs"]
        if resp_fs is None:
            assert data.get("_metadata/signals/gt_resp/has_timestamps")
            gt_resp_timestamps = data.get_t("gt_resp")
            assert gt_resp_full.shape == gt_resp_timestamps.shape    
        
        expected_signal_duration = self._instructions["expected_signal_duration"] # in seconds #TODO: add in/read from data
        signal_length = ppg.shape[0] # in number of samples
        assert signal_length == 1 + fs * expected_signal_duration
        
        
        window_duration = self._instructions["window_duration"] # in seconds
        window_step_duration = self._instructions["window_step_duration"] # second
        window_size = window_duration * fs # in num data points
        window_step = window_step_duration * fs # in num data points
        num_windows = int((signal_length - window_size)//window_step + 1)
        gt_resp_idx = None # index pointing to the mid 
        # point of window: from where the ground truth respiratory rate will 
        # be used
        
        
        results = self.create_new_results_container()
        context = self.create_new_context()
        
        self.apply_preprocessing_on_whole_signal(data=data, context=context)
        proc = context["signal_processor"]
        
        for window_idx in range(num_windows):
            
            context["current_window"] = None # reset current window info
            offset = window_idx * window_step # index offset
            end_ = offset + window_size
            t_start = offset * 1/fs
            t_end = end_ * 1/fs
            t = ((offset + end_)//2) * 1/fs
            has_artifacts = None
            # CHECK artifacts 
            if self._instructions["exclude_artifacts"]:
                has_artifacts = proc.check_if_chunk_has_artifacts(
                    data, start_time=t_start, end_time=t_end,
                    signal_name=signal_name
                )
                if has_artifacts:
                    # ignore chunks with artifacts
                    continue
            
            if resp_fs is not None:
                gt_resp_idx = int(t * resp_fs)
                # ground truth respiratory rate
                gt_resp = gt_resp_full[gt_resp_idx]
            else:
                gt_resp_idx = -1 # it's a dummy value
                gt_resp = proc.extract_ground_truth_rr(
                    reference_rr=gt_resp_full, timestamps=gt_resp_timestamps,
                    t_start=t_start, t_end=t_end,
                    mode=self._instructions["ground_truth_mode"])

            if gt_resp is None: # due to possible anomaly
                continue # do not add estimates to results
                # proceed to process the current signal window only if 
                # ground truth value was obtained
            
            current_window_data = {"window_idx": window_idx,
                                "offset": offset,
                                "end_": end_,
                                "t": t,
                                "gt_idx": gt_resp_idx,
                                "gt": gt_resp,
                                "has_artifacts": has_artifacts}
            
            context["current_window"] = current_window_data
            results = self.add_current_window_info_to_results(results,
                                                    current_window_data)
            
            
            
            rr_results = self.process_one_signal_window(
                data, context, fs, offset, end_)
            
            self.append_results(results, rr_results)
        
        return results

    def add_current_window_info_to_results(self, results_container, current_window_data):
        
        
        for k in current_window_data:
            if k in results_container:
                results_container[k].append(current_window_data[k])
        #>>> results["window_idx"].append(window_idx)
        #>>> results["offset"].append(offset)
        #>>> results["end_"].append(end_)
        #>>> results["t"].append(t)
        #>>> results["gt_idx"].append(gt_resp_idx)
        #>>> results["gt"].append(gt_resp)
        
        return results_container

    def create_new_context(self):
        # TODO: make reusable objects shared for all the runs
        proc = PpgSignalProcessor({}) # TODO : use config/ factory
        pulse_detector = PulseDetector()
        model = PROCESSOR_FACTORY.get(self._config["model"])
        
        # context object for each sample
        context = {
            "signal_processor": proc,
            "pulse_detector": pulse_detector,
            "model": model,
            "current_window": None # this attribute must be created afresh
            # for every new window
        }
        
        return context
    
    def create_new_results_container(self):
        results = {
            "window_idx": [],
            "offset": [],
            "end_": [],
            "t": [], # time elapsed from the start (of the signal) in `s`
            # corresponding to gt_idx
            "gt_idx" : [],
            "gt" : [],
            "rr_est_riav": [],
            "rr_est_riiv": [],
            "rr_est_rifv": [],
            "rr_est_fused": [],
            "rr_est_fused_valid": [],
            "std_rr_est_fused": []
        }
        
        return results

    def append_results(self, results_container, new_results):
        last_number_of_records = None
        for k in new_results:
            if k in results_container:
                n = len(results_container[k])
                if last_number_of_records is None:
                    last_number_of_records = len(results_container[k])
                if last_number_of_records != n:
                    raise AssertionError(f"Number of records mismatch. "
                            f"expected {last_number_of_records} , but for "
                            f" key {k} the number of records is {n}")
                results_container[k].append(new_results[k])
            else:
                logger.warning(f"Key {k} was not expected")
        
        return results_container

    def process_one_signal_window(self, data, context, fs, offset, end_):
        
        proc, pulse_detector, model = context["signal_processor"],\
            context["pulse_detector"], context["model"]
        
        re_riav, re_riiv, re_rifv = self.extract_respiratory_signal(
            data, proc, pulse_detector, fs, offset, end_)
        
        rr_est_riav = model.estimate_respiratory_rate(
            re_riav, CONF_FEATURE_RESAMPLING_FREQ, detrend=False,
            window_type=self._config["window_type"])
        rr_est_riiv = model.estimate_respiratory_rate(
            re_riiv, CONF_FEATURE_RESAMPLING_FREQ, detrend=False,
            window_type=self._config["window_type"])
        rr_est_rifv = model.estimate_respiratory_rate(
            re_rifv, CONF_FEATURE_RESAMPLING_FREQ, detrend=False,
            window_type=self._config["window_type"])
        
        logger.info([rr_est_riav, rr_est_riiv, rr_est_rifv])
        
        #>>> {
        #>>>     "rr_est_riav":  rr_est_riav,
        #>>>     "rr_est_riiv": rr_est_riiv,
        #>>>     "rr_est_rifv": rr_est_rifv
        #>>> }
        rr_est_fused, is_valid, rr_std = model.fuse_rr_estimates(
            rr_est_riav, rr_est_rifv, rr_est_riiv)
        
        results = {
            "rr_est_riav": rr_est_riav,
            "rr_est_riiv": rr_est_riiv,
            "rr_est_rifv": rr_est_rifv,
            "rr_est_fused": rr_est_fused,
            "rr_est_fused_valid": is_valid,
            "std_rr_est_fused": rr_std
        }
        return results

    def extract_respiratory_signal(self, data, proc, pulse_detector, fs, offset, end_):
        ppg = data.get("signals/ppg")
        filtered_signal = proc.eliminate_very_high_freq(ppg)
        signal_chunk = filtered_signal[offset:end_]
        new_peaklist, new_troughlist = pulse_detector.get_pulses(
            signal_chunk, fs)
        
        
        timesteps = data.get_t("ppg")[offset:end_]

        riav, riav_t = proc.extract_riav(signal_chunk, timesteps, new_peaklist,
                                         new_troughlist, None)
        rifv, rifv_t = proc.extract_rifv(signal_chunk, timesteps, new_peaklist,
                                         new_troughlist, None)
        riiv, riiv_t = proc.extract_riiv(signal_chunk, timesteps, new_peaklist,
                                         new_troughlist, None)
        
        resp_signals_and_times = [riav, riav_t, rifv, rifv_t, riiv, riiv_t]
        
        expected_length = np.ceil(
            (end_ - offset)*CONF_FEATURE_RESAMPLING_FREQ/fs).astype(int)
        re_riav, re_riiv, re_rifv = self.resample_resp_induced_signals(
            proc, resp_signals_and_times, expected_length)
                                             
        return re_riav,re_riiv,re_rifv

    def resample_resp_induced_signals(self, proc, resp_signals_and_times,
                                      expected_length):
        riav, riav_t, rifv, rifv_t, riiv, riiv_t = resp_signals_and_times
        re_riav, re_riav_t = proc.resample(riav, riav_t,
                                        CONF_FEATURE_RESAMPLING_FREQ, None)
        re_riiv, re_riiv_t = proc.resample(riiv, riiv_t,
                                        CONF_FEATURE_RESAMPLING_FREQ, None)
        re_rifv, re_rifv_t = proc.resample(rifv, rifv_t,
                                        CONF_FEATURE_RESAMPLING_FREQ, None)
                                        
        return re_riav,re_riiv,re_rifv
        
        
    def apply_preprocessing_on_whole_signal(self, data, context):
        """Transformations that are supposed to be done before the signal
        is processed window by window. Modifies `context`."""
        return context
    
    def accumulate_results(self, new_result, results_container):
        pass
    
    def summarize_results(self, result_container):
        pass


class Pipeline2(Pipeline):
    def __init__(self, config={}) -> None:
        """Pipeline to extract respiratory signals RIAV, RIFV and RIIV
        for the whole signal before analysing it window by window"""
        super().__init__(config)
    
    
    def process_one_signal_window(self, data, context, fs, offset, end_):
        pass
    
    def apply_preprocessing_on_whole_signal(self, data, context):
        
        ppg = data.get("signals/ppg")
        fs = data.get("_metadata/signals/ppg/fs")
        offset = 0
        end_ = ppg.shape[0] # end_ index in exclusive
        proc, pulse_detector, model = context["signal_processor"],\
            context["pulse_detector"], context["model"]
        re_riav,re_riiv,re_rifv = self.extract_respiratory_signal(
            data, proc, pulse_detector, fs, offset, end_)
        
        context["ppg_riav"] = re_riav
        context["ppg_rifv"] = re_rifv
        context["ppg_riiv"] = re_riiv
        
        return context
    
    def resample_resp_induced_signals(self, proc, resp_signals_and_times,
                                      expected_length):
        riav, riav_t, rifv, rifv_t, riiv, riiv_t = resp_signals_and_times
        re_riav, re_riav_t = proc.resample(riav, riav_t,
                                        None, expected_length)
        re_riiv, re_riiv_t = proc.resample(riiv, riiv_t,
                                        None, expected_length)
        re_rifv, re_rifv_t = proc.resample(rifv, rifv_t,
                                        None, expected_length)
                                        
        return re_riav,re_riiv,re_rifv
    
    def process_one_signal_window(self, data, context, fs, offset, end_):
        
        proc, pulse_detector, model = context["signal_processor"],\
            context["pulse_detector"], context["model"]
        
        re_riav, re_riiv, re_rifv = self._get_induced_signal_chunk(
            data, context, fs, offset, end_)
        
        rr_est_riav = model.estimate_respiratory_rate(
            re_riav, CONF_FEATURE_RESAMPLING_FREQ, detrend=False,
            window_type=self._config["window_type"])
        rr_est_riiv = model.estimate_respiratory_rate(
            re_riiv, CONF_FEATURE_RESAMPLING_FREQ, detrend=False,
            window_type=self._config["window_type"])
        rr_est_rifv = model.estimate_respiratory_rate(
            re_rifv, CONF_FEATURE_RESAMPLING_FREQ, detrend=False,
            window_type=self._config["window_type"])
        
        logger.info([rr_est_riav, rr_est_riiv, rr_est_rifv])
        
        
        rr_est_fused, is_valid, rr_std = model.fuse_rr_estimates(
            rr_est_riav, rr_est_rifv, rr_est_riiv)
        
        results = {
            "rr_est_riav": rr_est_riav,
            "rr_est_riiv": rr_est_riiv,
            "rr_est_rifv": rr_est_rifv,
            "rr_est_fused": rr_est_fused,
            "rr_est_fused_valid": is_valid,
            "rr_std": rr_std
        }
        return results
        
    def _get_induced_signal_chunk(self, data, context, fs, offset, end_):
        re_riav = context["ppg_riav"]
        re_rifv = context["ppg_rifv"]
        re_riiv = context["ppg_riiv"]
        
        idx_start = offset*(CONF_FEATURE_RESAMPLING_FREQ/fs)
        idx_end = end_*(CONF_FEATURE_RESAMPLING_FREQ/fs)
    
        idx_start = np.floor(idx_start).astype(int)
        idx_end = np.ceil(idx_end).astype(int)
        idx_start = max(0, idx_start)
        idx_end = min(idx_end, re_riav.shape[0])
        
        re_riav_chunk = re_riav[idx_start:idx_end]
        re_rifv_chunk = re_rifv[idx_start:idx_end]
        re_riiv_chunk = re_riiv[idx_start:idx_end]
        
        
        return re_riav_chunk, re_riiv_chunk, re_rifv_chunk
        

class DatasetBuilder(Pipeline2):
    
    def __init__(self, config={}) -> None:
        super().__init__(config)
    
    def process_one_signal_window(self, data, context, fs, offset, end_):
        
        proc, pulse_detector, model = context["signal_processor"],\
            context["pulse_detector"], context["model"]
        
        re_riav, re_riiv, re_rifv = self._get_induced_signal_chunk(
            data, context, fs, offset, end_)
        
        ppg_chunk = data.get("signals/ppg")[offset:end_]
        
        results = {
            "ppg_chunk": ppg_chunk,
            "riiv_chunk": re_riiv,
            "rifv_chunk": re_rifv,
            "riav_chunk": re_riav
        }
        return results
    
    def create_new_results_container(self):
        results = {
            "window_idx": [],
            "offset": [],
            "end_": [],
            "t": [], # time elapsed from the start (of the signal) in `s`
            # corresponding to gt_idx
            "gt_idx" : [],
            "gt" : [],
            "riav_chunk": [],
            "riiv_chunk": [],
            "rifv_chunk": [],
            "ppg_chunk": []
        }
        
        return results
    
    
    def close(self):
        self.save_dataset()
        super().close()
        
    def save_dataset(self):
        
        results = self._buffer["results"]
        errors = self._buffer["errors"]
        
        if len(errors) > 0:
            logger.error(f"Following samples had errors: {errors}")
        
        num_subs = len(results)
        sub_ids = []
        sub_idx = []
        y = [] # ground truth
        x = [] # signals
        for i in range(num_subs):
            r = results.pop()
            id_ = r["sample_id"]
            idx = r["idx"]
            v = r["output"]
            
            window_signals = v["ppg_chunk"]
            num_windows = len(window_signals)
            sub_ids = sub_ids + [id_] * num_windows
            sub_idx = sub_idx + [idx] * num_windows
            x = x + window_signals
            y = y + v["gt"] # ground truth
        
        df = pd.DataFrame(data=sub_idx, columns=["subject_idx"])
        df["subject_ids"] = sub_ids
        df["y"] = y
        x = np.stack(x, axis=0)
        x = pd.DataFrame(x, 
                         columns=["x_"+str(i).zfill(5) for i in range(x.shape[1])])
        df = pd.concat([x, df], axis=1)
        
        save_path = self.output_dir / "dataset.csv"
        df.to_csv(save_path)
        
        return df


class TrainingPipeline(BasePipeline):
    
    def __init__(self, config={}) -> None:
        super().__init__(config)
        self._instructions = self._config["instructions"]
        self._start_fold = self._config["dataloading"]["kwargs"]["config"]["start_fold"]
    
    def run(self, *args, **kwargs):
        logger.info("Starting")
        
        
        dataloader_composer = DATA_ADAPTER_FACTORY.get(
                                            self._config["dataloading"])
        start_fold = self._start_fold
        for fold_num in range(dataloader_composer.num_folds):
            fold = start_fold + fold_num
            logger.info(f"Running fold number {fold}")
            model = ML_FACTORY.get(self._config["model"])
            default_root_dir = self.output_dir / f"fold_{str(fold).zfill(2)}"
            
            if not self._instructions["do_only_test"]:
                ckpt_filename\
                    = "model-{epoch:02d}-s-{step}-{val_mae:.2f}"
                # TODO: add mode explicitly
                checkpoint_callback = ModelCheckpoint(
                    monitor="val_mae", save_top_k=4, filename=ckpt_filename)
                callbacks = [checkpoint_callback]
            else:
                assert self._instructions["ckpt_path"] != None
                callbacks = None
            train_loader, val_loader, test_loader \
                = dataloader_composer.get_data_loaders(current_fold=fold)
            trainer = pl.Trainer(default_root_dir=default_root_dir,
                                 callbacks=callbacks,
                                 **self._config["trainer"]["kwargs"])
            if not self._instructions["do_only_test"]:
                trainer.fit(model=model, train_dataloaders=train_loader,
                            val_dataloaders=val_loader)
                logger.info(f"Using best model@: {checkpoint_callback.best_model_path}")
                trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")
                
            else:
                trainer.test(model=model, dataloaders=test_loader, 
                             ckpt_path=self._instructions["ckpt_path"])
                
            
        
class IndexedDatasetBuilder(DatasetBuilder):
    """Creates a single output dataset `self.indexed_dataset`
    containing preprocessed signals (based on the provided signal processing
    object). Signal windows are not copied, rather only (dataset_id, sample_id,
    window_offset) are maintained along with one copy of the actual signal.
    Refer `self.create_indexed_dataset_struct` for detailed structure.
    The dataset created is supposed to be loaded as
    `ResprStandardIndexedDataContainer` which in turn can be used as the source
    to serve multiple torch.utils.data.Dataset by providing a copy of (partial
    or full ) index.
    """
    def __init__(self, config={}) -> None:
        super().__init__(config)
        self.indexed_dataset = self.create_indexed_dataset_struct()
        self.data_statistics = None # TODO: read from config - 
        # to be used for normalizing data etc.
        if "normalize_ppg" not in self._instructions:
            self._instructions["normalize_ppg"] = False
        
        
    def on_data_loaded(self, subject_ids, data_adapter):
        if self.data_statistics is None:
            self.data_statistics = self.compute_global_statistics(
                subject_ids, data_adapter)
        
        # Store data statistics in indexed_dataset
        self.indexed_dataset["datasets"][self._instructions["dataset_id"]]\
            ["data_statistics"] = copy.deepcopy(self.data_statistics)
        
    
    def compute_global_statistics(self, subject_ids, data_adapter):
        ppg = []
        gt_resp = []
        for idx, subject_id in enumerate(subject_ids):
            # if subject_id != "41": FIXME: investigate this
            
            
            logger.info(f"Processing subject#{subject_id}")
            data = data_adapter.get(subject_id)
            ppg.append(data.get("signals/ppg"))
            gt_resp.append(data.get("signals/gt_resp"))
        
        # concat arrays of shape (signal_length, )
        ppg = np.concatenate(ppg, axis=0) 
        # concat arrays of shape (num_gt, ): num_gt(number of ground truth
        # labels) may not stay fixed
        gt_resp = np.concatenate(gt_resp, axis=0)
        
        ppg_info = self.compute_stats(ppg)
        gt_info = self.compute_stats(gt_resp)
        
        data_statistics = {
            "ppg": ppg_info,
            "gt_resp": gt_info
        }
        logger.info(f"Updated self.data_statistics to: {self.data_statistics}")
        
        return data_statistics
        
        
    def compute_stats(self, a):
        
        if np.isnan(a).any():
            mean_func = np.nanmean
            std_func = np.nanstd
            med_func = np.nanmedian
            min_func = np.nanmin
            max_func = np.nanmax
        else:
            mean_func = np.mean
            std_func = np.std
            med_func = np.median
            min_func = np.min
            max_func = np.max
        
        if np.isinf(a).any():
            logger.warning(f"Array (shape={a.shape}) has inf values")
            a = a[np.isfinite(a)]
            logger.warning(f"Shape after removing infs: shape={a.shape}")
    
            
        
        mean = mean_func(a)
        std = std_func(a)
        med = med_func(a)
        min_ = min_func(a)
        max_ = max_func(a)
        
        return {
            "min": min_,
            "max": max_,
            "median": med,
            "std": std,
            "mean": mean
        }
    
        
        
    def create_indexed_dataset_struct(self):
        return {
            
            "index": [], # idx to (dataset_id, sample_id, sample_offset) map
            "_metadata": {
                "vector_length": self._instructions["vector_length"]
            },
            "dataset_index_to_id": {0: self._instructions["dataset_id"]},
            "dataset_id_to_index": {self._instructions["dataset_id"]: 0},
            "datasets": {
                
                # map of dataset_id : <dataset_obj>
                # NOTE: This class currently will add only
                # one `dataset` (i.e. just one entry in `datasets`)
                # <dataset_obj> looks like this:
                # { "dataset_id": <str>, 
                #   "data_statistics": {"x" : {"mean": <>, "min":  <>, 
                # "max": <>, "std": }, "y": {"mean": <>, ....}}
                #   "sample_ids": <list of sample ids> # to be used for partitioning
                #                initially it should be sorted. But later can be suffled.
                #   "samples": { 
                #           <sample_id> : {"x": <np.array> },
                #           <sample_id> : {"x": <np.array> },
                #           <sample_id> ...
                #            },
                #  "y": {
                #           <sample_id> : {<sample_offset>: <y_value>, <sample_offset>: <y_value> ....},
                #           <sample_id> : {...},
                #           <sample_id> ...
                # }
                self._instructions["dataset_id"]: {
                    "data_statistics": {},
                    "dataset_id": self._instructions["dataset_id"],
                    "sample_ids": [],
                    "samples": {},
                }
                
            }
        }
        

    
    
    def process_one_signal_window(self, data, context, fs, offset, end_):
        current_window_info = context["current_window"]
        ppg = data.get("signals/ppg")
        
        sample_id = data.get("id")
        
        y = current_window_info["gt"] # ground truth respiratory rate
        
        # DO normalization if requested
        if self._instructions["normalize_ppg"]:
            ppg = self.normalize_data(data_name="ppg", data=ppg)
        
        # adjust datat types
        ppg = np.array(ppg, dtype=DTYPE_FLOAT)
        y = np.array([y], dtype=DTYPE_FLOAT)
        
        if self._instructions["resample_ppg"]:
            if self._instructions["resampling_frequency"] != fs:
                # resample only if needed
                raise NotImplementedError()
        
        
        assert (end_ - offset ) == self._instructions["vector_length"]
        
        current_dataset_state = self.indexed_dataset["datasets"][
            self._instructions["dataset_id"]]
        
        
        if sample_id not in current_dataset_state["samples"]:
            current_dataset_state["samples"][sample_id] = {
                "x" : ppg,
                "y" : {}
            }
            current_dataset_state["sample_ids"].append(sample_id)
            #>>> old: current_dataset_state["y"][sample_id] = {}
        
        
        dataset_index = self.indexed_dataset["dataset_id_to_index"][
            self._instructions["dataset_id"]]
        
        
        
        # using offset as key in window to ground truth map
        #>>> old: current_dataset_state["y"][sample_id][offset] = y
        
        current_dataset_state["samples"][sample_id]["y"][offset] = y
        
        
        
        self.indexed_dataset["index"].append(
            (dataset_index, sample_id, offset)
        )
        
    def append_results(self, results_container, new_results):
        pass # not required for this pipeline
        
    def close(self):
        import pickle
        output_path = self.output_dir / \
            (self._instructions["dataset_id"] + ".pkl")
        logger.info(f"Saving indexed dataset at: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(self.indexed_dataset, f, 
                        protocol=pickle.HIGHEST_PROTOCOL)
    
    def normalize_data(self, data_name, data):
        """`data_name` is name of the signal/data. e.g. `ppg`, `gt_resp`"""
        stats = self.data_statistics[data_name]
        eps = 1e-7
        
        data = (data - stats["mean"])/(eps + stats["std"])
        
        return data
        
        
        
            
class DummyIndexedDatasetBuilder(IndexedDatasetBuilder):
    """ Creates dummy dataset for testing"""
    def __init__(self, config={}) -> None:
        super().__init__(config)
        self.num_dummy_datasets = 5
        self.num_dummy_samples_range = (5, 10) # in (a, b) both inclusive
        self.sample_id_prefixes = ["sample", "s", "smpl"]
        self.random_state = 0
        
    def run(self, *args, **kwargs):
        import random
        rn = random.Random(self.random_state)
        self.indexed_dataset = None
        v_len = self._instructions["vector_length"]
        d_id = self._instructions["dataset_id"]
        
        dataset_id_to_index = {}
        dataset_index_to_id = {}
        datasets = {}
        index = []
        for i in range(self.num_dummy_datasets):
            dataset_id = f"{d_id}_{str(i).zfill(4)}"
            dataset_id_to_index[dataset_id] = i
            dataset_index_to_id[i] = dataset_id
            num_samples = rn.randint(*self.num_dummy_samples_range)
            sample_id_prefix = rn.choice(self.sample_id_prefixes)
            
            sample_ids = []
            samples = {}
            
            for j in range(num_samples):
                sid = f"{sample_id_prefix}_{j}"
                offsets = rn.sample(range(1, 1000), rn.randint(5, 10))
                samples[sid] = {
                    "x": np.arange(1, 1000, 1),
                    "y": {i: i for i in offsets}
                }
                sample_ids.append(sid)
                index = index + [(i, sid, x_i) for x_i in offsets]
                
            datasets[dataset_id] = {
                "dataset_id": dataset_id,
                "sample_ids": sample_ids,
                "samples": samples,
            }
        self.indexed_dataset = {
            
            "index": index, # idx to (dataset_id, sample_id, sample_offset) map
            "_metadata": {
                "vector_length": v_len
            },
            "dataset_index_to_id": dataset_index_to_id,
            "dataset_id_to_index": dataset_id_to_index,
            "datasets": datasets
        }
        self.close()
    
            
        
        
            
        
REGISTERED_PIPELINES = {
    "Pipeline": Pipeline,
    "Pipeline2": Pipeline2,
    "DatasetBuilder": DatasetBuilder,
    "TrainingPipeline":TrainingPipeline,
    "IndexedDatasetBuilder": IndexedDatasetBuilder,
    "DummyIndexedDatasetBuilder": DummyIndexedDatasetBuilder
}  




