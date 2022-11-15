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


CONF_FEATURE_RESAMPLING_FREQ = 4 # Hz. (for riav/ rifv/ riiv resampling)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
class BasePipeline:
    
    def __init__(self, config={}) -> None:
        self._config = config
        # TODO: remove the following override
        self.creation_time = get_timestamp_str()
        self._config["output_dir"] = Path("../../artifacts")
        os.makedirs(self._config["output_dir"], exist_ok=True)
        
        self.root_output_dir = self._config["output_dir"]
        self.output_dir = self.root_output_dir / self.creation_time
        os.makedirs(self.output_dir, exist_ok=False)
        self._log_path = self.output_dir / f"{self.creation_time}_logs.log"
        self._log_sink = logger.add(self._log_path)
        
        
    
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
        
        
        self._config["window_type"] = "hamming"
        self._instructions = self._config["instructions"]
        
        cutl.save_yaml(self._config, self.output_dir/"config.yaml")
    
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
            
        
        # save results
        output_file = self.output_dir/ \
            ("output_" + get_timestamp_str() + ".pkl")
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        
        # close the pipeline properly
        self.close()
        
        
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
        
        expected_signal_duration = 8*60 # in seconds #TODO: add in/read from data
        signal_length = ppg.shape[0] # in number of samples
        assert signal_length == 1 + fs * expected_signal_duration
        
        
        window_duration = 32 # in seconds
        window_step_duration = 1 # second
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
            
            
            offset = window_idx * window_step # index offset
            end_ = offset + window_size
            t_start = offset * 1/fs
            t_end = end_ * 1/fs
            t = ((offset + end_)//2) * 1/fs
            
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
                    t_start=t_start, t_end=t_end)

            rr_results = self.process_one_signal_window(
                data, context, fs, offset, end_)
            
            if gt_resp is None: # due to possible anomaly
                continue # do not add estimates to results
            results["window_idx"].append(window_idx)
            results["offset"].append(offset)
            results["end_"].append(end_)
            results["t"].append(t)
            results["gt_idx"].append(gt_resp_idx)
            results["gt"].append(gt_resp)
            self.append_results(results, rr_results)
        
        return results

    def create_new_context(self):
        # TODO: make reusable objects shared for all the runs
        proc = PpgSignalProcessor({}) # TODO : use config/ factory
        pulse_detector = PulseDetector()
        model = PROCESSOR_FACTORY.get(self._config["model"])
        
        context = {
            "signal_processor": proc,
            "pulse_detector": pulse_detector,
            "model": model
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
            "rr_fused": [],
            "rr_fused_valid": [],
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
        rr_fused, is_valid = model.fuse_rr_estimates(
            rr_est_riav, rr_est_rifv, rr_est_riiv)
        
        results = {
            "rr_est_riav": rr_est_riav,
            "rr_est_riiv": rr_est_riiv,
            "rr_est_rifv": rr_est_rifv,
            "rr_fused": rr_fused,
            "rr_fused_valid": is_valid
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
        
        
        rr_fused, is_valid = model.fuse_rr_estimates(
            rr_est_riav, rr_est_rifv, rr_est_riiv)
        
        results = {
            "rr_est_riav": rr_est_riav,
            "rr_est_riiv": rr_est_riiv,
            "rr_est_rifv": rr_est_rifv,
            "rr_fused": rr_fused,
            "rr_fused_valid": is_valid
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
        
        
        
    
class DatasetBuilder:
    def __init__(self, config={}) -> None:
        self._config = config
        self._config["output_dir"] = Path("../../artifacts")
        os.makedirs(self._config["output_dir"], exist_ok=True)
        self.buffer = {
            "x": [],
            "y": [],
            "id": []
        }
        self.max_buffer_size = 100
        self.expected_feature_size = None
        self.zero_padded_record_count = 0
    
    def run(self, *args, **kwargs):
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
            
        # open store
        output_path = self._config["output_dir"] / \
            f"dataset_{get_timestamp_str()}.csv"

        for idx, subject_id in enumerate(subject_ids):

            logger.info(f"Processing subject#{subject_id}")
            data = bidmc_data_adapter.get(subject_id)
            try:
                self.process_one_sample(data)
                
            except KeyboardInterrupt:
                logger.error(traceback.format_exc())
                errors.append({
                    "idx": idx,
                    "sample_id": subject_id
                })
                continue
        
        if len(errors) > 0:
            logger.error(errors)
            
        # save data
        self.clean_and_close(output_path)
        # save results
        output_file = self._config["output_dir"] / \
            ("output_" + get_timestamp_str() + ".pkl")
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        
        
    def process_one_sample(self, data):
        proc = PpgSignalProcessor({}) # TODO : use config/ factory
        

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
        
        expected_signal_duration = 8*60 # in seconds #TODO: add in/read from data
        signal_length = ppg.shape[0] # in number of samples
        assert signal_length == 1 + fs * expected_signal_duration
        
        
        window_duration = 32 # in seconds
        window_step_duration = 16 # second
        window_size = window_duration * fs # in num data points
        window_step = window_step_duration * fs # in num data points
        num_windows = int((signal_length - window_size)//window_step + 1)
        gt_resp_idx = None # index pointing to the mid 
        # point of window: from where the ground truth respiratory rate will 
        # be used
        expected_feature_size = window_size
        
        if self.expected_feature_size is None:
            self.expected_feature_size = expected_feature_size

        
        for window_idx in range(num_windows):
            
            
            offset = window_idx * window_step # index offset
            end_ = offset + window_size
            t_start = offset * 1/fs
            t_end = end_ * 1/fs
            t = ((offset + end_)//2) * 1/fs
            
            signal_chunk = ppg[offset:end_]
            
            if resp_fs is not None:
                gt_resp_idx = int(t * resp_fs)
                # ground truth respiratory rate
                gt_resp = gt_resp_full[gt_resp_idx]
            else:
                gt_resp_idx = -1 # it's a dummy value
                gt_resp = proc.extract_ground_truth_rr(
                    reference_rr=gt_resp_full, timestamps=gt_resp_timestamps,
                    t_start=t_start, t_end=t_end)
            
            if gt_resp is None: # due to possible anomaly
                continue # do not add estimates to results
            
            padded_signal_chunk = self.zero_pad(signal_chunk,
                                                (self.expected_feature_size, ))
            padded_signal_chunk = np.expand_dims(padded_signal_chunk, 0)
            record = {
                "x": padded_signal_chunk,
                "y": gt_resp,
                "id": data.get("id")
            }
            self.add_record(record)
            
    def zero_pad(self, signal_chunk, expected_shape):
        padded = np.zeros(expected_shape)
        # FIXME: make efficient
        pad_width = expected_shape[0] - signal_chunk.shape[0]
        if pad_width == 0:
            return signal_chunk
        self.zero_padded_record_count += 1
        if signal_chunk.shape[0] > expected_shape[0]:
            return signal_chunk[0:expected_shape[0]]
        signal_chunk = np.pad(signal_chunk, (pad_width, 0), 
                              'constant', constant_values=(0, ))
        
        return signal_chunk
    def add_record(self, record):
        for k in record:
            self.buffer[k].append(record[k])
        # if len(self.buffer["y"]) == 5:
        #     self.clean_and_close("test.csv")
        
        
    
    def clean_and_close(self, output_path):
        x = np.concatenate(self.buffer["x"], axis=0)
        y = np.array(self.buffer["y"])
        y = np.reshape(y, (len(y), 1))
        id_ = np.array(self.buffer["id"])
        id_ = np.reshape(id_, (len(id_), 1))
        
        logger.info(f"Zeor padded records: {self.zero_padded_record_count}")
        logger.info(f"X shape: {x.shape}")
        
        
        
        df = pd.DataFrame(x, columns=[f"x_{i}" for i in range(x.shape[1])])
        
        # x_path = Path(str(output_path) + "_x.csv")
        df["y"] = y
        df["id_"] = id_
        df.to_csv(output_path, index=True)
        logger.info("Done")
        
        
REGISTERED_PIPELINES = {
    "Pipeline": Pipeline,
    "Pipeline2": Pipeline2,
    "DatasetBuilder": DatasetBuilder
}  

if __name__ == "__main__":
    import yaml
    DEFAULT_PIPELINE = "Pipeline2"
    config_path = DEFAULT_CONFIG_PATH
    config_data = None
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    pipeline_name = config_data["pipeline"]["name"]
    if pipeline_name is None:
        pipeline_name = DEFAULT_PIPELINE
    pipeline_class = REGISTERED_PIPELINES[pipeline_name]
    p = pipeline_class(config_data)
    p.run()
