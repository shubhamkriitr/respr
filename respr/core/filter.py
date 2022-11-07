import scipy
import matplotlib.pyplot as plt
import numpy as np

def create_fir(stop1, start1, start2, stop2, fs, lp=False):
    nyq_rate = fs / 2.0
    width = (stop2 - start2) / nyq_rate

    # attenuation in the stop band
    ripple_db = 60.0

    # order and Kaiser parameter
    N, beta = scipy.signal.kaiserord(ripple_db, width)

    if lp:
        taps = scipy.signal.firwin(N, stop2 / nyq_rate, window=('kaiser', beta), pass_zero=True)
    else:
        taps = scipy.signal.firwin(N, [stop1 / nyq_rate, stop2 / nyq_rate], window=('kaiser', beta), pass_zero=False)
    
    return (taps, 1.0)

def viz_filter(b, a, fs, xl=-1, new_fig=True):
    if new_fig:
        plt.subplots(figsize=(15,5))
    w, h = scipy.signal.freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h))

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    if xl > 0:
        plt.xlim([0, xl])
