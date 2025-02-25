from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from learn import Learn

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