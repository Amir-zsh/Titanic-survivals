from sklearn import ensemble
from DataManipulate import get_manipulated_train
def main():
    (X_train,X_test,Y_train,Y_test)=get_manipulated_train()
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    score=0
    for i in range(20):
        rf.fit(X_train,Y_train)
        score+=rf.score(X_test,Y_test)
    print(score/20)


if __name__ == "__main__":
    main()