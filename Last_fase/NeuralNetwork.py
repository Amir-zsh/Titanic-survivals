from DataRefine import get_refined_data,get_test_data
import numpy as np
import neurolab as nl
import pandas as pd

def make_result(out):
    result=[]
    for row in out:
        if row[0]>0.5:
            result.append(1)
        else:
            result.append(0)
    return result


def get_score(main,guessed):
    score=0
    size=len(guessed)
    for i in range(size):
        if main[i]==guessed[i]:
            score+=1
    return float(float(score)/float(size))



def predict():
    (X_train,X_test,Y_train,Y_test)=get_refined_data()
    boundaries=nl.tool.minmax(X_train)
    size=len(Y_train)
    target=np.reshape(Y_train,(size,1),1)
    net = nl.net.newff(boundaries,[5, 1])
    net.trainf=nl.train.train_rprop
    score=0
    error = net.train(X_train, target, epochs=1000, show=200)
    out = net.sim(X_test)
    result=make_result(out)
    score+=get_score(result,Y_test)
    print(score)
    # for i in range(10):
    #     error = net.train(X_train, target, epochs=1000, show=200)
    #     out = net.sim(X_test)
    #     result=make_result(out)
    #     score+=get_score(result,Y_test)
    # print(score/20)

def predict_test():
    (X_train,Y_train,X_test)=get_test_data()
    boundaries=nl.tool.minmax(X_train)
    size=len(Y_train)
    target=np.reshape(Y_train,(size,1),1)
    net = nl.net.newff(boundaries,[50, 1])
    net.trainf=nl.train.train_rprop
    error = net.train(X_train, target, epochs=100, show=200)
    out = net.sim(X_test)
    results=make_result(out)
    output = np.column_stack((X_test[:,0],results))
    df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
    df_results.to_csv('submit/titanic_results_NN.csv',index=False)



if __name__=="__main__":
    predict()