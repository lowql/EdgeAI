from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from load import WESAD
import pandas as pd
import numpy as np
from abc import abstractmethod

class Learn:
    def __init__(self, classifier, X:pd.DataFrame, y:pd.Series):
        self.X = X
        self.subjects = self.X['subject']
        self.y = X['label']
        self.X = X.drop(['label','subject'],axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self._clf = classifier
        self.model = classifier.fit(self.X_train,self.y_train)

    def check_test_XY(self) -> None:
        print("="*30)
        print(f"X: {self.X_test}")
        print(f"Y: {self.y_test}")
        print("="*30)

    def predict(self, testX:pd.DataFrame) -> None:
        predict = self.model.predict(testX)
        return predict
    
    def predict_proba(self, testX:pd.DataFrame) -> None:
        predict_proba = self.model.predict_proba(testX)
        return predict_proba

class Evaluate:
    # Base class for evaluating result
    def __init__(self, learn:Learn):
        self.learn = learn

    @abstractmethod
    def test(self):
        pass

class NormalEvaluate(Evaluate):
    # Evaluate trained model using pre-splitted test set
    def __init__(self, learn:Learn):
        super().__init__(learn)

    def test(self) -> None:
        report = classification_report(
            y_true=self.learn.y_test,
            y_pred=self.learn.predict(self.learn.X_test)
        )
        print(report)

class LOGOEvaluate(Evaluate):
    # Evaluate cross-validation untrained model LOGO using pre-splitted test set
    def __init__(self, learn:Learn): 
        self.logo = LeaveOneGroupOut()
        self.subjects = learn.subjects
        self.feature = learn.X
        self.labels = learn.y
        super().__init__(learn)

    def test(self) -> None:
        print(f"shape of [subject,feature,labels] {self.subjects.shape, self.feature.shape, self.labels.shape}")
        self._report()

    def _report(self) -> None:
        model = self.learn._clf
        
        self.feature = np.asarray(self.feature)
        y_pred = cross_val_predict(model, self.feature, self.labels, cv=self.logo, groups=self.subjects)
        acc = accuracy_score(self.labels, y_pred)
        print(f"model accuracy: {acc: .4f}")
        
        report = classification_report(self.labels, y_pred, digits=4)
        print(report)

    def _custom(self) -> None: 
        y_true, y_pred = [], []
        for train_idx, test_idx in self.logo.split(self.feature, self.labels, self.subjects):
            X_train, X_test = self.feature.iloc[train_idx], self.feature.iloc[test_idx]
            y_train, y_test = self.labels[train_idx], self.labels[test_idx]
            
            model = clone(self.learn._clf)
            model = model.fit(X_train, y_train)  # 重新训练模型
            y_pred.extend(model.predict(X_test))  # 收集预测结果
            y_true.extend(y_test)  # 收集真实标签

        # 计算准确率
        acc = accuracy_score(y_true, y_pred)
        print(f"Model accuracy: {acc:.4f}")
        
        report = classification_report(y_true, y_pred, digits=4)
        print(report)