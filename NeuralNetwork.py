from DataManipulate import get_manipulated_test,get_manipulated_train
import numpy as np
import neurolab as nl

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



def main():
    (X_train,X_test,Y_train,Y_test)=get_manipulated_train()
    boundaries=nl.tool.minmax(X_train)
    size=len(Y_train)
    target=np.reshape(Y_train,(size,1),1)
    net = nl.net.newff(boundaries,[50, 1])
    net.trainf=nl.train.train_rprop
    score=0
    error = net.train(X_train, target, epochs=20, show=200)
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


if __name__=="__main__":
    main()