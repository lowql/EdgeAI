import os,pickle
import pandas as pd
import nest_asyncio
import asyncio
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
    def mutiT_feature_extraction(self,sample_n=4000,window_size=3000,cols=['ECG','EMG','label'],limit=0):
        signal = self.group(sample_n=sample_n).loc[:,cols]
        rolling_windows = {key: self.rolling_window(signal[key],window_size=window_size) for key in cols}
        signal_length = limit if limit > 0 else len(self.rolling_window(signal['ECG'], window_size=window_size))
        print(f"signal length: {signal_length}")
        
        process_window_func = partial(
            self._process_window,
            rolling_windows = rolling_windows,
            cols = cols
        ) 
        features = []
        with ThreadPoolExecutor() as exec:
            features = list(exec.map(process_window_func,range(signal_length)))
        feat_df = pd.DataFrame(features)
        return feat_df
    def _process_window(self,i,rolling_windows,cols):
        col_feature = {}  # 先存這一組 window 的特徵  
        print("enter window")
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
        print(col_feature)
        print('exit window')
        return col_feature
# Use asyncio.run() in the correct context
if __name__ == '__main__':
    wesad = WESAD()
    print(wesad.df)