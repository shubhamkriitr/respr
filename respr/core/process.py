import numpy as np
import scipy.io
from scipy.signal import butter
from scipy import stats

# config keys
CONF_ELIM_VHF = "eliminate_vhf"
CONF_FS = "sampling_freq"
CONF_CUTOFF_HIGH = "cutoff_high"



class BasePpgSignalProcessor:
    
    def __init__(self, config):
        
        # ppg.Fpass = 38.5;  % in HZ
        # ppg.Fstop = 33.12;  % in HZ   
        # (33.12 and 38.5 provide a -3 dB cutoff of 35 Hz)
        # ppg.Dpass = 0.05;
        # ppg.Dstop = 0.01;
        # TODO: remove this config override
        config = {
            CONF_ELIM_VHF:{
                CONF_CUTOFF_HIGH: 35,
                CONF_FS: 125
            }
        }
        
        self._config = config

    
    def eliminate_very_high_freq(self, signal_, sampling_freq=None):
        # TODO: Use Kaiser Window Approach
        if sampling_freq is None:
            sampling_freq = self._config[CONF_ELIM_VHF][CONF_FS]
        
        # Lowpass filtering
        cutoff_high = self._config[CONF_ELIM_VHF][CONF_CUTOFF_HIGH]
        
        # Compute Wn units (N.B. Wn for Nyquist frequency = 1)
        wn = cutoff_high / sampling_freq * 2
        
        [b, a] = butter(1, wn,
                        btype='lowpass', output="ba", analog=False)

        signal_ = np.reshape(signal_, len(signal_))
        signal_ = scipy.signal.filtfilt(b, a, np.double(signal_))
        return signal_
    
    def clip_signal(self, signal_, k_upper, k_lower=None):
        if k_lower is None: k_lower = k_upper
        mean = np.mean(signal_)
        std = np.std(signal_)
        upper_limit = mean + std * k_upper
        lower_limit = mean - std * k_lower
        signal_ = np.clip(signal_, lower_limit, upper_limit)
        return signal_
    
    def filter_signal_(self, signal_, cutoff_band, **kwargs):
        pass


class PpgSignalProcessor(BasePpgSignalProcessor):
    
    def __init__(self, config):
        super().__init__(config)
    
    
    def extract_riiv(self, signal, timestep, peaklist, troughlist, sampling_freq):
        pass

class MultiparameterSmartFusion(object):
    
    
    def extract_riiv(self, signal, timestep, peaklist, troughlist, sampling_freq):
        pass
        
    
    def extract_riav(self):
        pass
    
    def extract_rifv(self):
        pass
    
    def aggregate_predictions(self):
        pass
    
    def create_respiratory_signal(self):
        pass
    
    def estimate_respiratory_rate(self):
        pass
    

if __name__ == "__main__":
    ppgproc = PpgSignalProcessor()
    
        