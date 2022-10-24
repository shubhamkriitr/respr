import numpy as np
import scipy.io
from scipy.signal import butter
from scipy import stats
from respr.util import logger

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


class PpgSignalProcessor(object):
    
    def __init__(self, config):
        # super().__init__(config)
        pass
        
    
    def _peak_trough_format_check(self, peaklist, troughlist):
        if len(peaklist) != len(troughlist):
            raise ValueError(f"Number of peaks({len(peaklist)}) do not mathch"
                             f"the number of troughs {len(troughlist)}")
    
    
    def extract_riav(self, signal, timestep, peaklist, troughlist, 
                     sampling_freq):
        """Extract respiratory induced amplitude variatiuons
            Args:
                peaklist: (Array)
        """
        
        self._peak_trough_format_check(peaklist, troughlist)
        # difference between peak and trough values
        riav = signal[peaklist] - signal[troughlist]
        riav_t = (timestep[peaklist] + timestep[troughlist])/2.0
        
        
        return riav, riav_t
    
    def extract_riiv(self, signal, timestep, peaklist, troughlist, 
                     sampling_freq):
        """Extract respiratory induced intensity variatiuons
            Args:
                peaklist: (Array)
        """
        
        self._peak_trough_format_check(peaklist, troughlist)
        
        # difference between peak and trough values
        riiv = signal[peaklist]
        riiv_t = timestep[peaklist]
        
        return riiv, riiv_t
    
    def extract_rifv(self, signal, timestep, peaklist, troughlist, 
                     sampling_freq):
        """Extract respiratory induced frequency variatiuons
            Args:
                peaklist: (Array)
        """
        self._peak_trough_format_check(peaklist, troughlist)
        t_peak = timestep[peaklist]
        rifv = np.diff(t_peak, n=1, axis=0)
        rifv_t = np.array([(t_peak[i]+t_peak[i-1])/2.0 
                           for i in range(1, len(t_peak))])
        return rifv, rifv_t
    
    
    def resample(self, signal_data, timesteps, output_sampling_freq):
        time_interval = timesteps[-1] - timesteps[0]
        num_points = round(time_interval * output_sampling_freq) + 1
        logger.debug(f"num_points={num_points} / freq: {output_sampling_freq}")
        resampled_signal, new_timesteps = scipy.signal.resample(
                                            x=signal_data, num=num_points,
                                            t=timesteps)
        return resampled_signal, new_timesteps

class MultiparameterSmartFusion(object):

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
    
        