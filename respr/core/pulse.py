import heartpy as hp
import numpy as np

class PulseDetector:
    
    def get_pulses(self, signal_to_process, sampling_freq):
        valid_peaks = self.get_peaks(signal_to_process, sampling_freq)
        valid_troughs = self.get_troughs(signal_to_process, sampling_freq)
        
        new_peaklist, new_troughlist = \
            self.adjust_peaks_and_troughs(valid_peaks, valid_troughs,
                                          signal_to_process)
            
        return new_peaklist, new_troughlist
        
    
    
    def get_peaks(self, signal_to_process, sampling_freq):
        working_data, measures = hp.process(signal_to_process, sampling_freq)
        valid_peaks = np.array(working_data["peaklist"])\
            [working_data["binary_peaklist"]==1]
        return valid_peaks
    
 
    def get_troughs(self, signal_to_process, sampling_freq):
        # invert 
        signal_to_process = -signal_to_process + np.max(signal_to_process)
        troughs = self.get_peaks(signal_to_process, sampling_freq)
        return troughs
    
    def adjust_peaks_and_troughs(self, peaklist, troughlist, signal_value):
    
        # fix leading troughs
        idx_t = 0
        min_trough_t = None
        min_trough_value = np.max(signal_value)
        new_troughlist = []
        new_peaklist = peaklist
        first_p1_t = peaklist[0] # time step(relative) at which first peak is detected
        discard_first_peak = False
        
        while idx_t < troughlist.shape[0]:
            tr_t =  troughlist[idx_t] # time step (relative and not absolute time) at which trough is detected
            
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
            
            p1_t, p2_t = peaklist[idx_p1], peaklist[idx_p1+1] # time steps at which peaks are detected
            
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
        
        
    
                                 

