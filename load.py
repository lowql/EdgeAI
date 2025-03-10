import os,pickle
import pandas as pd
import nest_asyncio
import asyncio
import matplotlib.pyplot as plt
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
nest_asyncio.apply() # IPython 環境中已經有 Event loop 需要做特殊處裡
from  pandas.core.series import Series
from collections import Counter
import feature as feat
def pickle_path(subject_path):
    """ Get dataset path """
    BASE_PATH = os.path.join("WESAD")
    SUBJECT_PATH = os.path.join(BASE_PATH, subject_path)
    return SUBJECT_PATH

def pickle_load_sync(pickle_path):
    """同步函數：讀取 pickle 檔案"""
    print(f"Start pickle: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    print(f"End pickle: {pickle_path}")
    return data
        
class WESAD:
    def __init__(self):
        self.df = pd.DataFrame()
        self.label = None
        self.subjects = list(range(2, 18))
        self.subjects.remove(12)  # Remove subject 12
        asyncio.run(self.build_df())  # 正确
        self.group_df = self.group().drop(columns=['label'])
    @classmethod
    def rolling_window(cls,feature:Series,window_size=10):
        if len(feature.tolist()) < window_size:
            raise IndexError(f"window size 大於 feature 的最大長度\n當前的feature長度為: {len(feature.tolist())}")
        return [feature[i:i+window_size].tolist() for i in range(len(feature)-window_size+1)]
    async def load_subject_data(self, subject_number):
        """Load data for a specific subject"""
        subject_path = f"S{subject_number}"
        full_path = os.path.join(pickle_path(subject_path), f"{subject_path}.pkl")
        
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as executor:
            data = await loop.run_in_executor(executor, pickle_load_sync, full_path)
            return self.pickle_to_df(data)
    def pickle_to_df(self, data):    
        label = data[b'label']
        data = data[b'signal']
        data = data[b'chest']
        data = {
            'label': label,
            **{f"ACC_{i}":x for i,x in enumerate(data[b'ACC'].T)},
            "ECG": data[b'ECG'][:,0],
            "EMG": data[b'EMG'][:,0], 
            "EDA": data[b'EDA'][:,0], 
            "Resp": data[b'Resp'][:,0], 
            "Temp": data[b'Temp'][:,0], 
        }     
        return pd.DataFrame(data)
    async def build_df(self):
        """Build DataFrame by loading data from all subjects"""
        # Use asyncio.gather to load data concurrently
        tasks = [self.load_subject_data(subject) for subject in self.subjects]
        subject_dataframes = await asyncio.gather(*tasks)
        
        # Concatenate all subject dataframes
        self.df = pd.concat(subject_dataframes, ignore_index=True)
        print("Finished building DataFrame")
        return self.df
    def group(self,sample_n=5):
        df = self.df[(self.df['label']==1) | (self.df['label']==2)] #「label」:1=基線（baseline），2=壓力（stress）
        df = df.groupby(['label']).apply(lambda x:x.sample(n=sample_n,random_state=42)).reset_index(drop=True) #Sample 40 from label==1 & label==2
        self.label = df['label']
        return df
    def feature_extraction(self,sample_n=4000,window_size=3000,cols=['ECG','EMG','label'],limit=0):
        signal = self.group(sample_n=sample_n).loc[:,cols]
        features = []
        signal_length = limit if limit > 0 else len(self.rolling_window(signal['ECG'], window_size=window_size))
        print(f"signal length: {signal_length}")
        for i in range(signal_length):  
            col_feature = {}  # 先存這一組 window 的特徵  

            for key in cols:  
                rolling_data = self.rolling_window(signal[key], window_size=3000)  
                time_signal = rolling_data[i]  # 取出對應索引的窗口資料  
                decorator = feat.SignalDecorator(time_signal)  

                if key == 'label':
                    col_feature['label'] = Counter(time_signal).most_common(1)[0][0]
                elif key == 'ECG':  
                    decorator.add_processor(feat.ButterBandpass(), lowcut=10, highcut=30, fs=70)  
                    decorator.add_processor(feat.HRVFrequency(), sampling_rate=700)  
                    processed_signal, results = decorator.apply()  
                    col_feature.update({f"ECG_{feat}": results['hrv_features'][feat] for feat in ['ULF', 'LF', 'HF', 'UHF']})   
                else:  
                    decorator.add_processor(feat.StdProcessor())  
                    processed_signal, results = decorator.apply()  
                    col_feature[f"std_{key}"] = processed_signal  
                    
                    decorator.add_processor(feat.MeanProcessor())  
                    processed_signal, results = decorator.apply()  
                    col_feature[f"mean_{key}"] = processed_signal  
                    

            features.append(col_feature)  

        feat_df = pd.DataFrame(features)
        return feat_df
    def mutiT_feature_extraction(self,sample_n=4000,window_size=3000,cols=['ECG','EMG','label'],limit=0,work_n=1):
        signal = self.group(sample_n=sample_n).loc[:,cols]
        rolling_windows = {key: self.rolling_window(signal[key],window_size=window_size) for key in cols}
        max_len_of_signal = len(self.rolling_window(signal['ECG'], window_size=window_size))
        signal_length = limit if limit > 0 else max_len_of_signal
        print(f"signal length: {signal_length}/{max_len_of_signal}")
        
        process_window_func = partial(
            self._process_window,
            rolling_windows = rolling_windows,
            cols = cols
        ) 
        features = []
        print(f"multi thread use {work_n} works")
        with ThreadPoolExecutor(max_workers=work_n) as exec:
            features = list(exec.map(process_window_func,range(signal_length)))
        feat_df = pd.DataFrame(features)
        return feat_df
    def _process_window(self,i,rolling_windows,cols):
        col_feature = {}  # 先存這一組 window 的特徵  
        for key in cols:  
            time_signal = rolling_windows[key][i]  # 取出對應索引的窗口資料  
            decorator = feat.SignalDecorator(time_signal)  
            if key == 'label':
                col_feature['label'] = Counter(time_signal).most_common(1)[0][0]
            elif key == 'ECG':  
                decorator.add_processor(feat.ButterBandpass(), lowcut=10, highcut=30, fs=70)  
                decorator.add_processor(feat.HRVFrequency(), sampling_rate=700)  
                processed_signal, results = decorator.apply()  
                col_feature.update({f"ECG_{feat}": results['hrv_features'][feat] for feat in ['ULF', 'LF', 'HF', 'UHF']})   
            else:  
                decorator.add_processor(feat.StdProcessor())  
                processed_signal, results = decorator.apply()  
                col_feature[f"std_{key}"] = processed_signal  
                
                decorator.add_processor(feat.MeanProcessor())  
                processed_signal, results = decorator.apply()  
                col_feature[f"mean_{key}"] = processed_signal 
        return col_feature

class Evaluate:
    plt.figure(figsize=(10, 6))
    @classmethod
    def with_row_limit(cls,wesad,limit_values):
        # 存儲結果的字典
        results = {
            'limit': [],
            'feature_extraction_time': [],
            'mutiT_feature_extraction_time': []
        }
        
        for limit in limit_values:
            # 測量第一個函數的執行時間
            start_time = time.time()
            wesad.feature_extraction(limit=limit)
            fe_time = time.time() - start_time
            
            # 測量第二個函數的執行時間
            start_time = time.time()
            wesad.mutiT_feature_extraction(limit=limit)
            mutiT_time = time.time() - start_time
            
            # 儲存結果
            results['limit'].append(limit)
            results['feature_extraction_time'].append(fe_time)
            results['mutiT_feature_extraction_time'].append(mutiT_time)
            print(results)
        
        
        cls.performance_data =  pd.DataFrame(results)
        plt.plot(cls.performance_data['limit'], cls.performance_data['feature_extraction_time'], 'o-', label='feature_extraction'),
        plt.plot(cls.performance_data['limit'], cls.performance_data['mutiT_feature_extraction_time'], 's-', label='mutiT_feature_extraction')
    
        return cls.performance_data
    @classmethod
    def with_diff_work(cls,wesad,work_n_s,limit=100):
        results = {'time':[],'works':[]}
        for work_n in work_n_s:
            # 測量第二個函數的執行時間
            start_time = time.time()
            wesad.mutiT_feature_extraction(limit=limit,work_n=work_n)
            mutiT_time = time.time() - start_time
            results['works'].append(work_n)
            results['time'].append(mutiT_time)
        cls.performance_data =  pd.DataFrame(results)
        plt.plot(cls.performance_data['works'], cls.performance_data['time'], 'o-', label='mutiT_feature_extraction with diff works'),
        return cls.performance_data
    @classmethod
    def show(cls):
        # 繪製性能比較圖
        if cls.performance_data.empty:
            raise ValueError("請先執行 with_XXX_XXX")
            
        plt.xlabel('limit')
        plt.ylabel('calculate cost (s)')
        plt.title('Comparison of performance')
        plt.legend()
        plt.grid(True)
        # plt.xscale('log')  # 使用對數刻度以更好地顯示不同量級的 limit
        plt.tight_layout()

        # 顯示結果表格
        print(cls.performance_data)

        # 顯示圖表
        plt.show()
        
# Use asyncio.run() in the correct context
if __name__ == '__main__':
    wesad = WESAD()
    print(wesad.df)