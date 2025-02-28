from load import WESAD
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        wesad = WESAD()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.df = wesad.get_df()
    def process(self):
        pass
    def show(self):
        # 在分組繪圖後加入這些指令
        # 方法一：使用 matplotlib 分開繪製
        plt.xlabel('EDA value')
        plt.ylabel('Frequence')
        plt.title('diff EDA of labels')
        plt.grid(True)
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 0:
            plt.legend()
        plt.show()  # 顯示圖表
        
# "Cat"代表類別型(Categorical)，"Con"代表連續型(Continuous)
class CatConPlotter(Plotter):
    def process(self,processor,**args):
        try:
            self.df_show = processor(self.df,**args)
        except Exception as e:
            print(e)
        return super().process()            

def processor(df,**args):
    match args['kind']:
        case 'hist':
            print("hist processor")
            return df.groupby('label')['EDA'].plot(**args)
        case 'box':
            print("box processer")
            return df.boxplot(column='EDA',by='label')
    raise Exception("Sorry, please check your plot kind") 

if __name__ == "__main__":
    catcon_plotter = CatConPlotter()
    
    hist_args = {'kind': 'hist', 'bins': 300, 'alpha': 0.5}
    catcon_plotter.process(processor,**hist_args)
    catcon_plotter.show()
    
    box_args = {'kind':'box'}
    catcon_plotter.process(processor,**box_args)
    catcon_plotter.show()
    