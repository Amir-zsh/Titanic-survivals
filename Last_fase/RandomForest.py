from sklearn import ensemble
# from DataManipulate import get_manipulated_train
from DataRefine import get_refined_data,get_test_data
import pandas as pd
import numpy as np
def predict():
    (X_train,X_test,Y_train,Y_test)=get_refined_data()
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    score=0
    for i in range(20):
        rf.fit(X_train,Y_train)
        score+=rf.score(X_test,Y_test)
    print(score/20)

def predict_test():
    (X_train,Y_train,X_test)=get_test_data()
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    rf.fit(X_train,Y_train)
    print "np.inf=", np.where(np.isnan(X_test))
    # print(X_test[152][0])
    results = rf.predict(X_test)
    output = np.column_stack((X_test[:,0],results))
    df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
    df_results.to_csv('submit/titanic_results_random_forest.csv',index=False)




if __name__ == "__main__":
    # print("random 1")
    predict()
    pass