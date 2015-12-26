from sklearn import ensemble
from DataManipulate import get_manipulated_train
def main():
    (X_train,X_test,Y_train,Y_test)=get_manipulated_train()
    gbt = ensemble.GradientBoostingClassifier()
    gbt.fit(X_test,Y_test)
    score=gbt.score(X_test,Y_test)
    print(score)
if __name__ == "__main__":
    main()