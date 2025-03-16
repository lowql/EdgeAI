from scipy.signal import butter, find_peaks,filtfilt,detrend,windows,welch
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

class SignalProcessor:
    def __call__(self,signal,**kwargs):
        raise NotImplementedError

class StdProcessor(SignalProcessor):
    def std(self,X,ddof=0):
        X = np.asarray(X)
        return np.std(X,ddof=ddof)
    def __call__(self, signal, **kwargs):
        return self.std(signal,**kwargs)
class MeanProcessor(SignalProcessor):
    def mean(self,X,axis=0):
        X = np.asarray(X)
        return np.mean(X,axis=axis)
    def __call__(self, signal, **kwargs):
        return self.mean(signal, **kwargs)
class MinMaxProcessor(SignalProcessor):
    def min_max(self,x,axis = None):
        x = np.asarray(x)
        min = x.min(axis=axis,keepdims=True)
        max = x.max(axis=axis,keepdims=True)
        return (x-min)/(max-min)
    def __call__(self, signal, **kwargs):
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
    def __call__(self, signal, **kwargs):
        return self.time_to_freq(signal,**kwargs)
class ButterBandpass(SignalProcessor):
    @classmethod
    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
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
        # print("nyq {} lowcut {} highcut {}".format(nyq,lowcut,highcut))
        if lowcut >= nyq or highcut >= nyq:
            raise ValueError(f"Cutoff frequencies must be less than Nyquist frequency ({nyq} Hz)")
        
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def __call__(self, signal, **kwargs):
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
class HRVFrequency(SignalProcessor):

    def __init__(self):
        self.bandpass_filter = ButterBandpass().butter_bandpass_filter
        super().__init__()
    @classmethod
    def extract_hrv_frequency_features(self, ecg_signal, sampling_rate=250):
        """
        Extract frequency domain HRV features from ECG signal
        
        Parameters:
        -----------
        ecg_signal : np.array
            The raw ECG signal
        sampling_rate : int
            Sampling rate of the ECG signal in Hz
            
        Returns:
        --------
        dict
            Dictionary containing ULF, VLF, LF, HF, and UHF energy components
        """
        freq_bands = {
            'ULF': (0.01, 0.04),
            'LF': (0.04, 0.15),
            'HF': (0.15, 0.4),
            'UHF': (0.4, 1.0)
        }
        # Step 1: ECG processing - R-peak detection
        # ecg_filtered = self.bandpass_filter(ecg_signal, lowcut=5, highcut=15, fs=sampling_rate)
        ecg_filtered = ecg_signal
        rpeaks, _ = find_peaks(ecg_filtered, height=0.5*np.max(ecg_filtered), distance=0.5*sampling_rate)
        
        # Step 2: Calculate RR intervals in seconds
        rr_intervals = np.diff(rpeaks) / sampling_rate
        
        # Optional: Remove outliers (ectopic beats)
        # rr_intervals = remove_outliers(rr_intervals)
        
        # Step 3: Interpolate to create evenly sampled signal
        # Interpolation frequency (typically 4 Hz for HRV analysis)
        fs_interp = 4.0
        
        # Create time array (cumulative sum of RR intervals)
        time_rr = np.cumsum(rr_intervals)
        time_rr = np.insert(time_rr, 0, 0)  # Insert 0 at beginning
        
        # Create evenly spaced time array for interpolation
        time_interp = np.arange(0, time_rr[-1], 1/fs_interp)
        
        # Interpolate RR intervals
        # print(f"RR intervals count: {len(rr_intervals)}, Time RR count: {len(time_rr)}")
        if len(rr_intervals) < 4:
            raise ValueError("Too few RR intervals for interpolation. Need at least 4.")
        if len(set(time_rr[:-1])) != len(time_rr[:-1]):
            raise ValueError("Duplicate time points found in time_rr.")

        tck = interpolate.splrep(time_rr[:-1], rr_intervals, s=0)
        rr_interpolated = interpolate.splev(time_interp, tck, der=0)
        
        # Step 4: Detrend the interpolated RR intervals
        # 移除數據的線性趨勢線
        rr_detrended = detrend(rr_interpolated)
        
        # Step 5: Apply windowing (e.g., Hann window)
        window = windows.hann(len(rr_detrended))
        rr_windowed = rr_detrended * window
        
        # Step 6: Compute power spectral density using Welch's method
        # Using Welch's method for better frequency resolution
        # Welch 方法的底層仍然使用 FFT，但它透過 分段、加窗、重疊與平均 來改善單次 FFT 可能遇到的問題。
        freqs, psd = welch(rr_windowed, fs=fs_interp, nperseg=len(rr_windowed)//2, 
                                scaling='density', detrend=False)
        
        # Step 7: Calculate energy in frequency bands
        # ULF, LF, HF, UHF => [[0.01, 0.04], [0.04, 0.15], [0.15, 0.4], [0.4, 1.0]]
        # Calculate energy in each band
        energies = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            # Find indices corresponding to the frequency band
            indices = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            # Calculate energy (area under the PSD curve) in the band
            band_energy = np.trapz(psd[indices], freqs[indices])
            energies[band_name] = band_energy
        
        # Calculate total power
        total_power = np.sum(list(energies.values()))
        
        # Calculate normalized powers and add to dictionary
        # 修正：使用原始字典的副本進行迭代，避免在迭代時修改字典
        
    
        for band in list(energies.keys()):
            if total_power == 0 or np.isnan(total_power):
                energies[f"{band}_normalized"] = np.nan  # 或者设为 0，取决于业务逻辑
            else:
                energies[f"{band}_normalized"] = energies[band] / total_power
        
        # Add LF/HF ratio
        energies['LF_HF_ratio'] = energies['LF'] / energies['HF'] if energies['HF'] > 0 else np.nan
        
        # return {"hrv_features":energies,"freq":freqs,"psd":psd}
        return [energies[feat] for feat in ['ULF', 'LF', 'HF', 'UHF']]

    def __call__(self, signal, **kwargs):
        return self.extract_hrv_frequency_features(signal,**kwargs)
    

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
            # print(processed_signal)
            # print(kwargs)
            result = processor.process(processed_signal, **kwargs)
            if isinstance(result, dict):
                results.update(result)  # 儲存結果
            else:
                processed_signal = result  # 更新處理後的信號
        self.processors = []
        return processed_signal, results

from typing import List, Callable
class FunctionPipeline:
    def __init__(self, process:List[Callable], kwargs:List[dict]) -> None:
        if len(process) != len(kwargs):
            raise AssertionError
        self.processors = [(func, args) for func, args in zip(process, kwargs)]
    
    def add_processor(self, processor:SignalProcessor, **kwargs:dict) -> None:
        self.processors.append((processor, kwargs))

    def apply(self, data: List) -> List[float]:
        result = data
        for func, args in self.processors:
            result = func(data, args)
        return result

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
