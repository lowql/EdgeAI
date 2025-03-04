import os,pickle
import pandas as pd
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
q = asyncio.Queue()
q2 = queue.Queue()
# pickle_lock = threading.Lock()
# C:\Users\vemi\Desktop\WESAD
def pickle_path(subject_path,pickle_filename):
    """ 取得資料集路徑 """
    BASE_PATH = os.path.join("WESAD")
    SUBJECT_PATH = os.path.join(BASE_PATH,subject_path)
    PICKLE = os.path.join(SUBJECT_PATH,pickle_filename)
    return PICKLE

def pickle_load_sync(pickle_path):
    """同步函數：讀取 pickle 檔案"""
    print(f"Start pickle: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    print(f"End pickle: {pickle_path}")
    return data

async def open_pickle(pickle_path):
    """非同步函數：使用多核心加速 pickle 讀取"""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        data = await loop.run_in_executor(executor, pickle_load_sync, pickle_path)  
        await q.put(data)  # 把結果放入非同步 Queue
        
class WESAD:
    def __init__(self):
        self._df = pd.DataFrame()
        self._label = None
        # 觸發非同步執行
        asyncio.create_task(self.run())  # 正確觸發非同步執行
    async def run(self):
        await self.build_df()

    def get_df(self):
        return self._df
    def get_label(self):
        return self._label
    def get_group_df(self):
        return self.group()
    def pickle_to_df(self, data):    
        label = data[b'label']
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
            'label': label,
            **{f"ACC_{i}":x for i,x in enumerate(data[b'ACC'].T)},
            "ECG": data[b'ECG'][:,0],
            "EMG": data[b'EMG'][:,0], 
            "EDA": data[b'EDA'][:,0], 
            "Resp": data[b'Resp'][:,0], 
            "Temp": data[b'Temp'][:,0], 
        }     
        # self.data = data
        q2.put(data)
    async def build_df(self):
        """ 建立DataFrame 準備訓練出所需要的資料欄位 """
        print("load: ",end='')
        tasks = []
        for i in range(2,18):
            if i == 12:
                continue
            print(f"S{i}",end=' ')
            path = pickle_path(f"S{i}",f"S{i}.pkl")
            tasks.append(open_pickle(path))
        await asyncio.gather(*tasks)
        
        while not q.empty():
            data = await q.get()
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
    