import os
from pathlib import Path
import yaml
import json
from respr.util import logger
from respr.util.common import (get_timestamp_str, PROJECT_ROOT)
import pickle
from respr.data import DATA_ADAPTER_FACTORY
from respr.data.bidmc import BidmcDataAdapter, BIDMC_DATSET_CSV_DIR
from respr.core.process import (PpgSignalProcessor, PROCESSOR_FACTORY)
from respr.core.pulse import PulseDetector
import heartpy as hp
import traceback

CONF_FEATURE_RESAMPLING_FREQ = 4 # Hz. (for riav/ rifv/ riiv resampling)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
class BasePipeline:
    
    def __init__(self, config={}) -> None:
        self._config = config
    
    def run(self, *args, **kwargs):
        pass
    
class Pipeline(BasePipeline):
    def __init__(self, config={}) -> None:
        super().__init__(config)
        # TODO: remove the following override
        self._config["output_dir"] = Path("../../artifacts")
        self._config["window_type"] = None #"hamming"
        os.makedirs(self._config["output_dir"], exist_ok=True)
    
    def run(self, *args, **kwargs):
        # bidmc_data_adapter = BidmcDataAdapter(
        #     {"data_root_dir": BIDMC_DATSET_CSV_DIR})
        bidmc_data_adapter = DATA_ADAPTER_FACTORY.get(
            self._config["data_adapter"])
        file_names = bidmc_data_adapter.inspect()
        subject_ids = bidmc_data_adapter.extract_subject_ids_from_file_names(file_names)
        
        
        results = []
        errors = []
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
            except KeyboardInterrupt:
                logger.error(traceback.format_exc())
                errors.append({
                    "idx": idx,
                    "sample_id": subject_id
                })
                continue
        
        if len(errors) > 0:
            logger.error(errors)
            
        
        # save results
        output_file = self._config["output_dir"] / \
            ("output_" + get_timestamp_str() + ".pkl")
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        
        
    def process_one_sample(self, data):
        proc = PpgSignalProcessor({}) # TODO : use config/ factory
        pulse_detector = PulseDetector()
        model = PROCESSOR_FACTORY.get(self._config["model"])
        
        # params
        #125 # Sampling freq. TODO extract from data
        fs = data.value()["_metadata"]["signals"]["ppg"]["fs"]
        fs = int(fs)
        
        # sampling freq of respiratory rate (in Hz.)
        resp_fs =  data.value()["_metadata"]["signals"]["gt_resp"]["fs"]
        ppg = data.value()["signals"]["ppg"]
        gt_resp_full = data.get("signals/gt_resp")
        
        expected_signal_duration = 8*60 # in seconds
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
            "rr_est_rifv": []
            
        }
        
        for window_idx in range(num_windows):
            
            
            offset = window_idx * window_step # index offset
            end_ = offset + window_size
            t = ((offset + end_)//2) * 1/fs
            gt_resp_idx = int(t * resp_fs)
            
            
            # ground truth respiratory rate
            gt_resp = gt_resp_full[gt_resp_idx]
            
            rr_est_riav, rr_est_riiv, rr_est_rifv = self.get_respiratory_rate(
                data, proc, pulse_detector, model, fs, offset, end_)
            
            results["window_idx"].append(window_idx)
            results["offset"].append(offset)
            results["end_"].append(end_)
            results["t"].append(t)
            results["gt_idx"].append(gt_resp_idx)
            results["gt"].append(gt_resp)
            results["rr_est_riav"].append(rr_est_riav)
            results["rr_est_riiv"].append(rr_est_riiv)
            results["rr_est_rifv"].append(rr_est_rifv)
        
        return results

    def get_respiratory_rate(self, data, proc, pulse_detector, model, fs, offset, end_):
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
        
        re_riav, re_riav_t = proc.resample(riav, riav_t,
                                             CONF_FEATURE_RESAMPLING_FREQ)
        re_riiv, re_riiv_t = proc.resample(riiv, riiv_t,
                                             CONF_FEATURE_RESAMPLING_FREQ)
        re_rifv, re_rifv_t = proc.resample(rifv, rifv_t,
                                             CONF_FEATURE_RESAMPLING_FREQ)
        
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
        return rr_est_riav, rr_est_riiv, rr_est_rifv
        
        
        
    
    def accumulate_results(self, new_result, results_container):
        pass
    
    def summarize_results(self, result_container):
        pass
    
    
if __name__ == "__main__":
    import yaml
    config_path = DEFAULT_CONFIG_PATH
    config_data = None
    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    logger.info(f"config = {config_data}")
    p = Pipeline(config_data)
    p.run()