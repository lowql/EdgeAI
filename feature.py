from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from  pandas.core.frame import DataFrame

def rolling_window(feature:DataFrame,window_size=10):
    return [feature[i:i+window_size].tolist() for i in range(len(feature)-window_size)]
        
class Transform:
    def __init__(self,time_signal):
        self.time_signal = time_signal
        self.idx = len(time_signal)
        self.idx_half = self.idx // 2
    def time_to_freq(self,time_step=0.01):
        """
            time_step=0.001: 假設取樣率為 100 Hz
        """
        # 計算 FFT
        self.freq_signal = np.fft.fft(self.time_signal)
        
        self.freqs = np.fft.fftfreq(self.idx,d=time_step)  
        self.freqs_half = self.freqs[:self.idx_half]
       
        self.magnitude_half = np.abs(self.freq_signal[:self.idx_half])
        self.magnitude = np.abs(self.freq_signal)
    def freq_pass_butterfly(self):
        porcessor = SignalProcess()
        self.time_signal = porcessor.butter_bandpass_filter(self.time_signal,lowcut=10,highcut=50,fs=200)
        return self.time_signal
    def plot(self,figsize=(10,6)):
        fig, ax = plt.subplots(4, 1, figsize=figsize)
        # 時域信號
        self.time_to_freq()
        ax[0].plot(self.time_signal, label="ECG Signal")
        ax[0].set_title("Time Domain Signal")# 時域信號
        ax[0].set_xlabel("time")# 時間點
        ax[0].set_ylabel("Amplitude")# 振幅
        ax[0].legend()
        
        # 頻域信號
        ax[1].stem(self.freqs_half, self.magnitude_half, 'b',basefmt=" ")
        ax[1].set_title("Frequency Domain Signal (FFT)")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Magnitude")
        
        # 頻域信號
        self.freq_pass_butterfly()
        self.time_to_freq()
        ax[2].plot(self.time_signal, label="ECG Signal")
        ax[2].set_title("Filter Time Domain Signal")# 時域信號
        ax[2].set_xlabel("time")# 時間點
        ax[2].set_ylabel("Amplitude")# 振幅
        ax[2].legend()

        
        # 頻域信號
        ax[3].stem(self.freqs_half, self.magnitude_half, 'b',basefmt=" ")
        ax[3].set_title("Filter Frequency Domain Signal (FFT)")
        ax[3].set_xlabel("Frequency (Hz)")
        ax[3].set_ylabel("Magnitude")
        plt.tight_layout()
        plt.show()

            

class SignalProcess:
    def butter_bandpass(self,lowcut:float, highcut:float, fs:int, order=5):
        """
        # from scipy.signal import butter, lfilter
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
        # Validate input frequencies
        if lowcut >= nyq or highcut >= nyq:
            raise ValueError(f"Cutoff frequencies must be less than Nyquist frequency ({nyq} Hz)")
        # print("nyq: {} low: {} high {}".format(nyq,low,high))
        
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
            energies.append(self.butter_bandpass_filter(sampled_signal, bands[0], bands[1], sampling_frequency, order))
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