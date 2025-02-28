from scipy.signal import butter, lfilter
import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt

def mock_signal():
    num_taps = 51 # it helps to use an odd number of taps
    cut_off = 3000 # Hz
    sample_rate = 32000 # Hz
    # create our low pass filter
    mock_signal = signal.firwin(num_taps, cut_off, fs=sample_rate)
    # plot the impulse response
    plt.plot(mock_signal, '.-')
    plt.show()
class SignalProcess:
    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        """
        Create butterworth bandpass filter
        Args:
            lowcut (float): low cut frequency
            highcut (float): high cut frequency
            fs (int): sampling rate
            order (int, optional): order of filter. Defaults to 5.
        Returns:
            b, a (np.ndarray): filter coefficients
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='band')
        return b, a
    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        """設置帶通濾波器參數，根據指定的低頻、高頻和階數進行濾波"""
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    def signal_frequency_band_energies(self,sampled_signal, frequency_bands, sampling_frequency, order=5):
        energies = []
        for bands in frequency_bands:
            energies.append(butter_bandpass_filter(sampled_signal, bands[0], bands[1], sampling_frequency, order))
        return energies
    def calculate_band_energy(self,signal, lowcut, highcut, fs, order=5):
        """計算濾波後信號的能量（平方和）"""
        filtered_signal = self.butter_bandpass_filter(signal, lowcut, highcut, fs, order)
        energy = np.sum(filtered_signal**2) 
        return energy

if __name__ == "__main__":
    signal_processer = SignalProcess()
    # ULF, LF, HF, UHF => [[0.01, 0.04], [0.04, 0.15], [0.15, 0.4], [0.4, 1.0]]
    energies = signal_processer.signal_frequency_band_energies(mock_signal, [[0.01, 0.04], [0.04, 0.15], [0.15, 0.4], [0.4, 1.0]], 32)