import os,pickle
import pandas as pd

def pickle_path(subject_path,pickle_filename):
    BASE_PATH = os.path.join(os.path.dirname(__file__),"WESAD")
    SUBJECT_PATH = os.path.join(BASE_PATH,subject_path)
    PICKLE = os.path.join(SUBJECT_PATH,pickle_filename)
    return PICKLE
def open_pickle(pickle_path):
    with open(pickle_path,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        return(data)
    
class WESAB:
    def __init__(self):
        self._df = None
        self._label = None
        s2_pickle = pickle_path("S2","S2.pkl")
        print(s2_pickle)
        data = open_pickle(s2_pickle)
        self.build_df(data)
    def get_df(self):
        return self._df
    def get_label(self):
        return self._label
    def get_group_df(self):
        return self.group()
    def build_df(self,data):
        self._label = data[b'label']
        data = data[b'signal']
        data = data[b'chest']
        data = {
            'label': self._label,
            **{f"ACC_{i}":x for i,x in enumerate(data[b'ACC'].T)},
            "ECG": data[b'ECG'][:,0], #.flatten() [:,0]
            "EMG": data[b'EMG'][:,0], 
            "EDA": data[b'EDA'][:,0], 
            "Resp": data[b'Resp'][:,0], 
            "Temp": data[b'Temp'][:,0], 
        }
        self.data = data
        self._df = pd.DataFrame(data)
    def group(self):
        df = self._df[(self._df['label']==1) | (self._df['label']==2)]
        df = df.groupby('label').apply(lambda x:x.sample(n=40,random_state=42)).reset_index(drop=True)
        self._label = df['label']
        df = df.drop(columns=['label'])
        return df
        
        
if __name__ == "__main__":
    wesab = WESAB()
    group_data = wesab.group()
    print(group_data)
    