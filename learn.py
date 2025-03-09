from sklearn.model_selection import train_test_split
from load import WESAD
wesab = WESAD()
class Learn:
    def __init__(self, clf):
        X = wesab.group_df
        y = wesab.label
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self._clf = clf.fit(self.X_train, self.y_train)
    def check_test_XY(self):
        print("="*30)
        print(f"X: {self.X_test}")
        print(f"Y: {self.y_test}")
        print("="*30)
    def test(self)-> str:
        from sklearn.metrics import classification_report
        report = classification_report(y_true=self.y_test,y_pred=self.predict(self.X_test))
        return report
    def predict(self,testX):
        predict = self._clf.predict(testX)
        return predict
    def predict_proba(self,testX):
        predict_proba = self._clf.predict_proba(testX)
        print(predict_proba)
        return predict_proba

    