import numpy as np
from dataclasses import dataclass
import scipy

SIGNAL_DTYPE = np.float64

@dataclass(frozen=True)
class DATAKEYS:
    HAS_ARTIFACTS = "has_artifacts"
    SIG = "signals"
    PPG = "ppg"
    TIME = "time"
    HAS_TS = "has_timestamps"
    META = "_metadata"
    AF_LOC = "artifacts_loc"
class BaseDataAdapter:
    
    def __init__(self, config):
        self.config = config
        self.data_root_dir = self.config["data_root_dir"]
        self.signal_dtype = SIGNAL_DTYPE
    
    def inspect(self):
        raise NotImplementedError()
    
    def extract_subject_ids_from_file_names(self, file_names:list):
        raise NotImplementedError()
    
    def get_file_paths(self, id_):
        raise NotImplementedError()
    
    def get(self, id_):
        raise NotImplementedError()
    
    def resample_ppg(self, data, num_points: int, f_resample: int):
        """Resamples ppg and modifies `data` in place.
        """
        assert isinstance(f_resample, int)
        ppg = data.get("signals/ppg")
        ppg_meta = data.get("_metadata/signals/ppg")
        assert ppg_meta["has_timestamps"]
        t_key = ppg_meta["t_loc"]
        assert t_key == "time/ppg"
        ppg_t = data.get(t_key)
        
        
        
        resampled_signal, new_timesteps = scipy.signal.resample(
                                            x=ppg, num=num_points,
                                            t=ppg_t)
        
        # change meta data
        ppg_meta["fs"] = f_resample
        ppg_meta["t_is_uniform"] = True
        
        data.value()["signals"]["ppg"] = resampled_signal
        data.value()["time"]["ppg"] = new_timesteps
        data.value()["_metadata"]["signals"]["ppg"] = ppg_meta
        
        return data
        
        
    

class StandardDataRecord:
    def __init__(self, data) -> None:
        self._data = data
        
    def value(self):
        return self._data
    
    def get_t(self, signal_name):
        t_loc = self.get(f"_metadata/signals/{signal_name}/t_loc")
        t = self.get(t_loc)
        return t
        
    
    def get(self, key = ""):
        value = self.extract(self._data, key)
        return value
    
    def extract(self, root, key):
        value = root
        for k in key.split("/"):
            value = value[k]
        return value

