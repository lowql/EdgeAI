import os,pickle
import pandas as pd
import nest_asyncio
import asyncio
from concurrent.futures import ProcessPoolExecutor
nest_asyncio.apply() # IPython 環境中已經有 Event loop 需要做特殊處裡

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
        self._df = pd.DataFrame()
        self._label = None
        self.subjects = list(range(2, 18))
        self.subjects.remove(12)  # Remove subject 12
        asyncio.run(self.build_df())  # 正确
    async def load_subject_data(self, subject_number):
        """Load data for a specific subject"""
        subject_path = f"S{subject_number}"
        full_path = os.path.join(pickle_path(subject_path), f"{subject_path}.pkl")
        
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as executor:
            data = await loop.run_in_executor(executor, pickle_load_sync, full_path)
            return self.pickle_to_df(data)

    @property
    def df(self):
        return self._df
    @property
    def label(self):
        return self._label
    @property
    def group_df(self):
        return self.group()
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
        self._df = pd.concat(subject_dataframes, ignore_index=True)
        print("Finished building DataFrame")
        return self._df
    def group(self):
        df = self._df[(self._df['label']==1) | (self._df['label']==2)] #「label」:1=基線（baseline），2=壓力（stress）
        df = df.groupby(['label']).apply(lambda x:x.sample(n=40,random_state=42)).reset_index(drop=True) #Sample 40 from label==1 & label==2
        self._label = df['label']
        df = df.drop(columns=['label'])
        return df
        
        
# Use asyncio.run() in the correct context
if __name__ == '__main__':
    wesad = WESAD()
    print(wesad.df)