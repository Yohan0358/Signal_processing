from scipy import signal
from scipy.signal import spectrogram

import numpy as np
import matplotlib.pyplot as plt

def FFT(wave, sr, window = 'hanning', 
           nperseg = 4096, overlap = 50, 
           nfft = 4096, scaling = 'spectrum', 
           mode = 'magnitude', ref = 2e-5,
           plot_type = '1D'):
    '''
    wave : signal data
    sr : sampling rate
    window : window function for each block
    nperseg : block size
    ref : reference for dB scale
          sound pressure : 2e-5
          vibration : 1e-6
    plot_type : 1D - FFT average
              : 2D - FFT vs time(waterfall)
    '''
    overlap = nperseg * 50 // 100
    freq, t, Spectrum_x = spectrogram(wave, 
                                        fs = sr, 
                                        window = window, 
                                        nperseg = nperseg, 
                                        nfft = nfft, 
                                        noverlap = 50, 
                                        scaling = scaling, 
                                        mode = mode)

    if plot_type == '1D':
        Sx = Spectrum_x.mean(axis = 1) ** 0.5 # Root mean square for each block
        plt.figure()
        plt.plot(freq, 20 * np.log10(Sx / ref))
        plt.xscale('log')
        plt.show()

    elif plot_type == '2D':
        plt.figure()
        plt.imshow(Spectrum_x, origin = 'lower', extent = [t.min(), t.max(), freq.min(), freq.max()],
                    interpolation = 'bilinear', cmap = 'jet', aspect = 'auto')
        plt.show()