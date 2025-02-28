from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from learn import Learn
""" 
## 決策樹 [(Decision Tree)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
根據資料特徵進行分割，形成樹狀結構。 (Gini impurtiy or Entropy )

## 隨機森林 [(Random Forest)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
Bagging 多棵決策樹的整合，通過投票決策，增強分類穩定性。
- 避免相較決策樹，避免過度擬和，減少分支域值的影響。

## [Adaboost Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
疊加多個簡單模型，增強分類效果。
- 會通過一定的迭代行為，根據不同分類器的錯誤率，決定分類器的權重
- 集成學習 ， 多預測器訓練

## 線性判別分析 [(LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
統計參數型的監督式學習
LDA希望投影後的資料，組內分散量(within-class scatter)越小越好，組間分散量(between-class scatter)越大越好。


## K近鄰 [(KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
根據最近的K個鄰居進行分類。
1. 距離計算：對於新樣本,計算它與訓練集中每個樣本的距離
2. 選擇近鄰：選出距離最近的K個樣本
3. 投票表決：這K個樣本中,哪個類別出現最多,就將新樣本歸為該類別
"""
if __name__ == "__main__":
    models = [DecisionTreeClassifier(random_state=0),
              RandomForestClassifier(),
              AdaBoostClassifier(n_estimators=100,random_state=0),
              LinearDiscriminantAnalysis(),
              KNeighborsClassifier(n_neighbors=3)]
    for model in models:
        class_name = model.__class__.__name__
        print(f"---------------{class_name}---------------")
        learn = Learn(model)
        print(learn.test())
        print('-'*(30+len(class_name)))