import os,pickle
import pandas as pd

def pickle_path(subject_path,pickle_filename):
    """ 取得資料集路徑 """
    BASE_PATH = os.path.join("WESAD")
    SUBJECT_PATH = os.path.join(BASE_PATH,subject_path)
    PICKLE = os.path.join(SUBJECT_PATH,pickle_filename)
    return PICKLE
def open_pickle(pickle_path):
    """ 取得資料級內容 """
    with open(pickle_path,'rb') as f:
        data = pickle.load(f,encoding='bytes',fix_imports=False)
        return data
    
class WESAD:
    def __init__(self):
        self._df = None
        self._label = None
        self.build_df()
    def get_df(self):
        return self._df
    def get_label(self):
        return self._label
    def get_group_df(self):
        return self.group()
    def build_df(self):
        """ 建立DataFrame 準備訓練出所需要的資料欄位 """
        self._df = pd.DataFrame()
        print("load: ",end='')
        for i in range(2,18):
            if i == 12:
                continue
            print(f"S{i}",end=' ')
            data = open_pickle(pickle_path(f"S{i}",f"S{i}.pkl"))
            self._label = data[b'label']
            data = data[b'signal']
            data = data[b'chest']
            """ 
            (ACC_1,ACC_2,ACC_3): 測量物體加速度，檢測姿勢與識別活動行為
            ECG: 心電圖，診斷心臟健康
            EDA: 皮膚電活動，反應交感神經活動水平，用於情感識別與壓力檢測
            EMG: 肌電圖
            RESP: 呼吸訊號(頻率、速度)
            TEMP: 體溫
            BVP: 血容量脈動(心律、心血管脈動)
            """
            data = {
                'label': self._label,
                **{f"ACC_{i}":x for i,x in enumerate(data[b'ACC'].T)},
                "ECG": data[b'ECG'][:,0],
                "EMG": data[b'EMG'][:,0], 
                "EDA": data[b'EDA'][:,0], 
                "Resp": data[b'Resp'][:,0], 
                "Temp": data[b'Temp'][:,0], 
            }
            self.data = data
            self._df = pd.concat([self._df, pd.DataFrame(data)], ignore_index=True)
        print("Finish build DataFrame")
    def group(self):
        df = self._df[(self._df['label']==1) | (self._df['label']==2)] #「label」:1=基線（baseline），2=壓力（stress）
        df = df.groupby(['label']).apply(lambda x:x.sample(n=40,random_state=42)).reset_index(drop=True) #Sample 40 from label==1 & label==2
        self._label = df['label']
        df = df.drop(columns=['label'])
        return df
        
        
if __name__ == "__main__":
    wesab = WESAD()
    df = wesab._df
    # group_data = wesab.group()
    # print(group_data)
    