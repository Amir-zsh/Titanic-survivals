from sklearn import ensemble
from DataRefine import get_refined_data,get_test_data
import numpy as np
import pandas as pd

def predict():
    (X_train,X_test,Y_train,Y_test)=get_refined_data()
    gbt = ensemble.GradientBoostingClassifier()
    gbt.fit(X_train,Y_train)
    score=gbt.score(X_test,Y_test)
    print(score)
    #############################################################
def predict_test():
    (X_train,Y_train,X_test)=get_test_data()
    gbt = ensemble.GradientBoostingClassifier()
    gbt.fit(X_train,Y_train)
    results = gbt.predict(X_test)
    output = np.column_stack((X_test[:,0],results))
    df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
    df_results.to_csv('submit/titanic_results_gradient_boosting.csv',index=False)

if __name__ == "__main__":
    predict()