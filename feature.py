from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from  pandas.core.frame import DataFrame
def rolling_window(feature:DataFrame,window_size=10):
    return [feature[i:i+window_size].tolist() for i in range(len(feature)-window_size+1)]

class SignalProcessor:
    def process(self,signal,**kwargs):
        raise NotImplementedError

class MinMaxProcessor(SignalProcessor):
    def min_max(self,x,axis = None):
        x = np.asarray(x)
        min = x.min(axis=axis,keepdims=True)
        max = x.max(axis=axis,keepdims=True)
        return (x-min)/(max-min)
    def process(self, signal, **kwargs):
        return self.min_max(signal,**kwargs)
class FFTProcessor(SignalProcessor):
    def time_to_freq(self,time_signal,time_step=0.01):
        """
            time_step=0.001: 假設取樣率為 100 Hz
        """
        signal_len = len(time_signal)
        idx,idx_half = signal_len,signal_len//2
        freq_signal = np.fft.fft(time_signal)
        freqs = np.fft.fftfreq(idx,d=time_step)  
        freqs_half = freqs[:idx_half]
        magnitude_half = np.abs(freq_signal[:idx_half])
        return {"freqs":freqs_half,"magnitude":magnitude_half}
    def process(self, signal, **kwargs):
        return self.time_to_freq(signal,**kwargs)

class ButterBandpass(SignalProcessor):
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
    
    def process(self, signal, **kwargs):
        return self.butter_bandpass_filter(signal,**kwargs)
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

class SignalDecorator:
    """裝飾器模式，允許對信號進行多重處理"""
    def __init__(self, signal):
        self.signal = signal
        self.processors = []

    def add_processor(self, processor:SignalProcessor, **kwargs):
        """新增信號處理策略"""
        self.processors.append((processor, kwargs))

    def apply(self):
        """依序應用處理策略"""
        processed_signal = self.signal
        results = {}
        for processor, kwargs in self.processors:
            result = processor.process(processed_signal, **kwargs)
            if isinstance(result, dict):
                results.update(result)  # 儲存結果
            else:
                processed_signal = result  # 更新處理後的信號
        return processed_signal, results
    
class SignalVisualizer:
    """負責繪圖"""
    def __init__(self, original_signal, time_step=0.01):
        self.original_signal = original_signal
        self.time_step = time_step

    def plot(self, processed_signal, results, figsize=(10,6)):
        fig, ax = plt.subplots(3, 1, figsize=figsize)

        # 原始信號
        ax[0].plot(self.original_signal, label="Original Signal")
        ax[0].set_title("Time Domain Signal")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Amplitude")
        ax[0].legend()

        # 頻域信號
        if "freqs" in results and "magnitude" in results:
            freqs_half = results["freqs"][:len(results["freqs"]) // 2]
            magnitude_half = results["magnitude"][:len(results["magnitude"]) // 2]
            ax[1].stem(freqs_half, magnitude_half, 'b', basefmt=" ")
            ax[1].set_title("Frequency Domain Signal (FFT)")
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_ylabel("Magnitude")

        # 處理後的信號
        ax[2].plot(processed_signal, label="Processed Signal")
        ax[2].set_title("Processed Time Domain Signal")
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("Amplitude")
        ax[2].legend()

        plt.tight_layout()
        plt.show()
            

        