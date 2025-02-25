# 事前準備
下載WESAD資料集: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx

# 機器學習演算法
## 決策樹 [(Decision Tree)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
根據資料特徵進行分割，形成樹狀結構。
## 隨機森林 [(Random Forest)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
多棵決策樹的整合，通過投票決策，增強分類穩定性。
## [Adaboost Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
疊加多個簡單模型，增強分類效果。
## 線性判別分析 [(LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
找到使類別間區分最大的投影方向。
## K近鄰 [(KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
根據最近的K個鄰居進行分類。

# HW1
請訓練模型能夠分類出兩個類型的情緒(中性、壓力)。 *「label」:1=基線（baseline），2=壓力（stress）
- 使用所有受試者的Chest資料(ACC、ECG、EMG、EDA、RESP、TEMP)
- 使用所有受試者的label 1 跟 2 各取40筆資料
- 不需要做額外的特徵處理
- 使用第6頁的五個演算法
- 輸出模型訓練後的準確度報告

Train_set:Test_set = 7:3

Train_set = 80 * (7/10) = 56
Test_set = 80 * (3/10) = 24
```shell
PS D:\_study\DataScience\EdgeAI> python -m main
D:\_study\DataScience\EdgeAI\WESAD\S2\S2.pkl
---------------DecisionTreeClassifier---------------
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        11

    accuracy                           1.00        24
   macro avg       1.00      1.00      1.00        24
weighted avg       1.00      1.00      1.00        24

----------------------------------------------------
---------------RandomForestClassifier---------------
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        11

    accuracy                           1.00        24
   macro avg       1.00      1.00      1.00        24
weighted avg       1.00      1.00      1.00        24

----------------------------------------------------
---------------AdaBoostClassifier---------------
              precision    recall  f1-score   support

           1       1.00      0.85      0.92        13
           2       0.85      1.00      0.92        11

    accuracy                           0.92        24
   macro avg       0.92      0.92      0.92        24
weighted avg       0.93      0.92      0.92        24

------------------------------------------------
---------------LinearDiscriminantAnalysis---------------
              precision    recall  f1-score   support

           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        11

    accuracy                           1.00        24
   macro avg       1.00      1.00      1.00        24
weighted avg       1.00      1.00      1.00        24

--------------------------------------------------------
---------------KNeighborsClassifier---------------
              precision    recall  f1-score   support

           1       1.00      0.85      0.92        13
           2       0.85      1.00      0.92        11

    accuracy                           0.92        24
   macro avg       0.92      0.92      0.92        24
weighted avg       0.93      0.92      0.92        24

--------------------------------------------------
```