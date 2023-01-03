import numpy as np
import scipy.io
from scipy.signal import butter
from scipy import stats
from respr.util import logger
from scipy.fft import fft, fftfreq
from respr.core.filter import create_fir
from respr.util.common import BaseFactory, fill_missing_values
# config keys
CONF_ELIM_VHF = "eliminate_vhf"
CONF_FS = "sampling_freq"
CONF_CUTOFF_HIGH = "cutoff_high"
from respr.data.base import StandardDataRecord, DATAKEYS


class BasePpgSignalProcessor:
    
    def __init__(self, config):
        
        # ppg.Fpass = 38.5;  % in HZ
        # ppg.Fstop = 33.12;  % in HZ   
        # (33.12 and 38.5 provide a -3 dB cutoff of 35 Hz)
        # ppg.Dpass = 0.05;
        # ppg.Dstop = 0.01;
        default_config = {
            CONF_ELIM_VHF:{
                CONF_CUTOFF_HIGH: 35,
                CONF_FS: 125
            }
        }
        
        self._config = config
        self._config = fill_missing_values(default_values=default_config,
                                           target_container=self._config)

    
    def eliminate_very_high_freq(self, signal_, sampling_freq=None):
        # TODO: Use Kaiser Window Approach
        if sampling_freq is None:
            sampling_freq = self._config[CONF_ELIM_VHF][CONF_FS]
        
        # Lowpass filtering
        cutoff_high = self._config[CONF_ELIM_VHF][CONF_CUTOFF_HIGH]
        
        # Compute Wn units (N.B. Wn for Nyquist frequency = 1)
        wn = cutoff_high / sampling_freq * 2
        
        [b, a] = butter(5, wn,
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
    
    
    def resample(self, signal_data, timesteps, output_sampling_freq, number_of_points):
        """_summary_

        Args:
            signal_data (_type_): _description_
            timesteps (_type_): _description_
            output_sampling_freq (_type_): Resampling frequency. (this argument
                is not used if `number_of_points` is provided)
            number_of_points (_type_): number of points expected in output.
                (It takes precedence over `output_sampling_freq`)

        Returns:
            _type_: _description_
        """
        if number_of_points is None:
            time_interval = timesteps[-1] - timesteps[0]
            num_points = round(time_interval * output_sampling_freq) + 1
            logger.debug(f"num_points={num_points} / freq: {output_sampling_freq}")
        else:
            num_points = number_of_points
        resampled_signal, new_timesteps = scipy.signal.resample(
                                            x=signal_data, num=num_points,
                                            t=timesteps)
        return resampled_signal, new_timesteps
    
    def adjust_peaks_and_troughs(self, peaklist, troughlist,
                                 timestep, signal_value):
        # fix leading troughs
        idx_t = 0
        min_trough_t = None
        min_trough_value = np.max(signal_value)
        new_troughlist = []
        new_peaklist = peaklist
        # time step(relative) at which first peak is detected
        first_p1_t = peaklist[0]
        discard_first_peak = False
        
        while idx_t < troughlist.shape[0]:
            tr_t =  troughlist[idx_t] # time step (relative and not absolute 
            # time) at which trough is detected
            
            if tr_t > first_p1_t:
                break
            
            tr_v = signal_value[tr_t] # signal value at the trough location
            
            if tr_v < min_trough_value:
                min_trough_value = tr_v
                min_trough_t = tr_t
            
            idx_t += 1
        
        if min_trough_t is not None:
            new_troughlist.append(min_trough_t)
        else:
            discard_first_peak = True
            
        
        # scan and fix remaining
        # idx_t is taken from before
        idx_p1 = 0
        tr_count = 0 # running count of troughs b/w two peaks
        min_tr_t = None
        min_tr_value = np.max(signal_value)
        
        while idx_p1 < (peaklist.shape[0]-1) :
            
            # time steps at which peaks are detected
            p1_t, p2_t = peaklist[idx_p1], peaklist[idx_p1+1] 
            
            while idx_t < troughlist.shape[0]:
                tr_t =  troughlist[idx_t] # time step at which trough is detected

                tr_v = signal_value[tr_t]


                if tr_t < p2_t:
                    if min_tr_value > tr_v:
                        min_tr_value = tr_v
                        min_tr_t = tr_t

                    idx_t += 1
                    tr_count += 1
                else:
                    break
                    
            if tr_count == 0:
                # select min location from signal
                min_tr_t = p1_t + np.argmin(signal_value[p1_t:p2_t+1])
            
            new_troughlist.append(min_tr_t)
            
            # prepare vars for next peak pair
            min_tr_value = np.max(signal_value)
            tr_count = 0
            min_tr_t = None
            idx_p1 += 1
        
            
            
        if discard_first_peak:
            new_peaklist = peaklist[1:]
        
        return new_peaklist, np.array(new_troughlist)
    
    def extract_ground_truth_rr(self, reference_rr, timestamps, 
                                t_start, t_end, mode="mean"):
        """Extract (compute) ground truth respiratory rate for a window of
        `reference_rr` (reference respiratory rate array). The window is 
        specified by [`t_start`, `t_end`) (N.B. t_end excluded). `timestamps` is
        the time corresponding to each of the values recorded in `reference_rr`
        #TODO: doc string
        Args:
            reference_signal (_type_): _description_
            timestamps (_type_): _description_
            t_start (_type_): _description_
            t_end (_type_): _description_
            mode (str, optional): Method to obtain the ground truth
                Defaults to "mean".

        Raises:
            
        Returns:
            _type_: _description_
        """
        start_idx = np.searchsorted(timestamps, t_start, side="left")
        end_idx = np.searchsorted(timestamps, t_end, side="right")
        
        
        y = reference_rr[start_idx:end_idx]
        
        if mode == "mean":
            y_overall = np.mean(y)
            
        elif mode == "mid":
            t_ = timestamps[start_idx:end_idx]
            y_overall = self.extract_ground_truth_rr_as_mid(y, t_)
        else:
            raise NotImplementedError()
        if y_overall is None:
            return None
        if np.isnan(y_overall):
            logger.warning(f"nan encountered : [{start_idx}, {end_idx})")
            return None
        if y_overall > 120:
            logger.warning(
                f"Possible anomaly: rr > 120: [{start_idx}, {end_idx})")
            return None
        # TODO: may also return aprox. timestamp
        return y_overall
    
    def extract_ground_truth_rr_as_mid(self, y_, t_):
        if y_.shape[0] == 0:
            return None
        if y_.shape[0] == 1:
            return y_[0]
        
        t_mid = (t_[0] + t_[-1])/2
        
        # takinf idx of timestamp closest to mid 
        mid_idx = np.argmin(np.abs(t_ - t_mid))
        #>>> other similar way: mid_idx = np.searchsorted(t_, t_mid, side="left")
        
        return y_[mid_idx]
    
    def check_if_chunk_has_artifacts(self, data: StandardDataRecord,
                                     start_time, end_time, signal_name="ppg"):
        # TODO: doc string
        """_summary_

        Args:
            data (StandardDataRecord): current signal data record
            start_time (float): start time of the window (chunk)
            end_time (float): end time of the window (chunk)

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        meta_key = f"{DATAKEYS.META}/{DATAKEYS.SIG}/{signal_name}"
        meta = data.get(meta_key)
        if DATAKEYS.HAS_ARTIFACTS in meta:
            if not meta[DATAKEYS.HAS_ARTIFACTS]:
                return False
        else:
            return False
                
        # case: signal has artifacts 
        # check if the queried chunk has artifacts
        artifact_key = data.get(f"{meta_key}/{DATAKEYS.AF_LOC}")
        artifact_times = data.get(artifact_key)
        
        assert artifact_times.shape[0] % 2 == 0 #even num. of elements expected
        
        # init
        artifact_start_times = [] # in sec
        artifact_end_times = [] # in sec
        has_artifact_in_this_chunk = False
        
        idx = 0

        while idx < artifact_times.shape[0]-1:
            current_start_time = artifact_times[idx]
            current_end_time = artifact_times[idx + 1]
            if not ((end_time < current_start_time) 
                    or (start_time > current_end_time)):
                artifact_start_times.append(current_start_time)
                artifact_end_times.append(current_end_time)
            idx += 2

        if len(artifact_start_times) > 0:
            has_artifact_in_this_chunk = True
            artifact_start_times[0] = max(artifact_start_times[0], start_time)
            artifact_end_times[-1] = min(artifact_end_times[-1], end_time)
            
            
        # TODO: may also compute and return fraction of the chunk 
        # that is artifact
        return has_artifact_in_this_chunk
        
class PpgSignalProcessorUpdateResample(PpgSignalProcessor):
    def __init__(self, config):
        super().__init__(config)
    
    def resample(self, signal_data, timesteps, output_sampling_freq, number_of_points):
        if number_of_points is None:
            time_interval = timesteps[-1] - timesteps[0]
            num_points = round(time_interval * output_sampling_freq) + 1
            logger.debug(f"num_points={num_points} / freq: {output_sampling_freq}")
        else:
            num_points = number_of_points
        
        _start = timesteps[0]
        _step = 1./output_sampling_freq
        _stop = _start + num_points*(1/output_sampling_freq) + _step
                    
        new_timesteps = np.arange(start=_start, stop=_stop,
                                  step=_step)
        f = scipy.interpolate.interp1d(x=timesteps, y=signal_data,
                                            kind="cubic")
        resampled_signal = f(new_timesteps)
        
        return resampled_signal, new_timesteps
        

from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class MultiparameterSmartFusion(object):
    
    def __init__(self, config):
        self._config = config
        defaults = {
            "butterworth_filter_order": 5
        }
        self._config = fill_missing_values(default_values=defaults,
                                           target_container=self._config)
        

    def extract_riav(self):
        pass
    
    def extract_rifv(self):
        pass
    
    def aggregate_predictions(self):
        pass
    
    def create_respiratory_signal(self):
        pass    

    def estimate_respiratory_rate(self, resp_signal, sampling_freq,
                                  detrend=True, elim_non_resp=True,
                                  window_type=None):
        N = resp_signal.shape[0]
        T = 1.0/ sampling_freq
        
        resp_signal = self.clean_signal_for_psd(
            resp_signal, sampling_freq, detrend, elim_non_resp, window_type)
        
        bpm = self.extract_bpm(resp_signal, N, T)
        #>>> plt.plot(x_fft_per_min, y_final)
        #>>> plt.scatter([bpm], [bpm_response])
        #>>> plt.grid()
        #>>> plt.show()
        return bpm

    def extract_bpm(self, resp_signal, N, T, pad=True):
        
        #>>> x = np.linspace(0.0, N*T, N, endpoint=False)
        if pad:
            pad_width_l = resp_signal.shape[0]
            
            pad_width_r = pad_width_l + pad_width_l % 2
            
            resp_signal = np.pad(resp_signal, (pad_width_l, pad_width_r), 
                              'constant', constant_values=(0, 0))
            N = resp_signal.shape[0]
            
        
        y_fft = fft(resp_signal)
        x_fft = fftfreq(N, T)[:N//2]

        print(f" N = {N}/ T = {T}/ / resp = {resp_signal.shape}")
        print(f" x_fft = {x_fft.shape}/ y_fft = {y_fft.shape}")
        
        x_fft_per_min = x_fft *  60
        y_final = 2.0/N * np.abs(y_fft[0:N//2])
        max_energy_idx = np.argmax(y_final)
        bpm, bpm_response = x_fft_per_min[max_energy_idx], y_final[max_energy_idx]
        return bpm

    def clean_signal_for_psd(self, resp_signal, sampling_freq, detrend,
                             elim_non_resp, window_type):
        N = resp_signal.shape[0]
        # detrend the signal
        if detrend:
            logger.debug("Detrend (mean)")
            resp_signal = scipy.signal.detrend(resp_signal, type="constant")
        
         # keep only plausible respiratory frequencies
        if elim_non_resp:
            resp_signal = self.eliminate_non_respiratory_frequencies(
                            resp_signal, sampling_freq)
        
        if window_type is not None:
            assert window_type in ["hamming"] # TODO: check if want to use other types of windows
            _window = scipy.signal.get_window(window_type, N)
            resp_signal = _window*resp_signal
        return resp_signal
        
    
    
    def eliminate_non_respiratory_frequencies(self,  signal_, sampling_freq=None):
        # TODO: Use Kaiser Window Approach
        if sampling_freq is None:
            raise NotImplementedError()

        # Bandpass filter signal
        cutoff_low = self._config["resp_rate_cutoff_low"]/60.0
        cutoff_high = self._config["resp_rate_cutoff_high"]/60.0
        
        
        filter_order = self._config["butterworth_filter_order"]
        
        [b, a] = butter(filter_order, [cutoff_low / sampling_freq * 2, cutoff_high / sampling_freq * 2],
                        btype='bandpass', analog=False)

        signal_ = np.reshape(signal_, len(signal_))
        signal_ = scipy.signal.filtfilt(b, a, np.double(signal_))

        return signal_
    
    def fuse_rr_estimates(self, rr_riav, rr_rifv, rr_riiv):
        rrs = np.array([rr_riav, rr_rifv, rr_riiv])
        stdv = np.std(rrs)
        rr_aggregated = np.mean(rrs)
        if stdv > 4.0: # stddev > 4 breaths/min
            return rr_aggregated, False, stdv # discard
        return rr_aggregated, True, stdv
class MultiparameterSmartFusion2(MultiparameterSmartFusion):
    def __init__(self, config):
        super().__init__(config)
        self._plot = False # For debugging only 
    
    def eliminate_non_respiratory_frequencies(self,  signal_, sampling_freq=None):
        logger.info("Skipping: eliminate_non_respiratory_frequencies ")
        # FIXME: adjust this function
        # return signal_
        # TODO: Use Kaiser Window Approach
        if sampling_freq is None:
            raise NotImplementedError()

        # Bandpass filter signalb
        cutoff_low = 2/60.0 #using 1bpm / TODO: add to config
        cutoff_high = 36/60.0 # using 36bpm / TODO: add to config
        cutoff_high_start = (36 - 1)/60.0
        
        [b, a] = create_fir(cutoff_low, None, cutoff_high_start, 
                            cutoff_high, fs=sampling_freq, lp=False)

        signal_ = np.reshape(signal_, len(signal_))
        signal_ = scipy.signal.lfilter(b, a, np.double(signal_))

        return signal_
    
    def extract_bpm(self, resp_signal, N, T):
        sampling_frequency = 1/ T
        f, psd = scipy.signal.welch(resp_signal, fs=sampling_frequency,
                                     nfft=200*sampling_frequency,
                                     nperseg=3*sampling_frequency,
                                     noverlap=2*sampling_frequency,
                                     detrend=False)

    
        
        max_energy_idx = np.argmax(psd)
        
        bpm = f[max_energy_idx] *  60
        
        # TODO: remove later
        if self._plot:
            plt.figure()
            plt.plot(f, psd)
            plt.xlabel("f [Hz]")
            plt.ylabel("PSD")
        
        
        return bpm
        

class PpgSignalProcessor2(PpgSignalProcessor):
    def __init__(self, config):
        super().__init__(config)
    def extract_riav(self, signal, timestep, peaklist, troughlist, sampling_freq):
        riav, riav_t = super().extract_riav(
            signal, timestep, peaklist, troughlist, sampling_freq)
        riav = riav/np.mean(riav)
        return riav, riav_t
    
    def extract_riiv(self, signal, timestep, peaklist, troughlist, sampling_freq):
        riiv, riiv_t = super().extract_riiv(
            signal, timestep, peaklist, troughlist, sampling_freq)
        riiv = riiv/np.mean(riiv)
        return riiv, riiv_t
    
    def extract_rifv(self, signal, timestep, peaklist, troughlist, sampling_freq):
        rifv, rifv_t = super().extract_rifv(
            signal, timestep, peaklist, troughlist, sampling_freq)
        rifv = rifv/np.mean(rifv)
        return rifv, rifv_t


COMPONENTS_MAP = {
    "MultiparameterSmartFusion2": MultiparameterSmartFusion2,
    "MultiparameterSmartFusion": MultiparameterSmartFusion,
    "PpgSignalProcessorUpdateResample": PpgSignalProcessorUpdateResample,
    "PpgSignalProcessor": PpgSignalProcessor,
    "PpgSignalProcessor2": PpgSignalProcessor2
}

PROCESSOR_FACTORY = BaseFactory({"resource_map" : COMPONENTS_MAP})
if __name__ == "__main__":
    ppgproc = PpgSignalProcessor()
    
        