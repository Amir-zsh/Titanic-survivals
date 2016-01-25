import pandas as pd
import numpy as np
from sklearn import cross_validation

def get_manipulated_train():
    df = pd.read_csv('data/train.csv',header=0)
    columns = ['Name','Ticket','Cabin']
    df = df.drop(columns,axis=1)
    redundant = []
    columns = ['Pclass','Sex','Embarked']
    for col in columns:
        redundant.append(pd.get_dummies(df[col]))
    titanic_redundant = pd.concat(redundant, axis=1)
    df = pd.concat((df,titanic_redundant),axis=1)
    df = df.drop(['Pclass','Sex','Embarked'],axis=1)
    df['Age'] = df['Age'].interpolate()
    Y = df['Survived'].values
    X = df.values
    X = np.delete(X,1,axis=1)
    # return (X,Y)
    return  cross_validation.train_test_split(X,Y,test_size=0.3,random_state=0)

def get_manipulated_test():
    df = pd.read_csv('data/test.csv',header=0)
    columns = ['Name','Ticket','Cabin']
    df = df.drop(columns,axis=1)
    redundant = []
    columns = ['Pclass','Sex','Embarked']
    for col in columns:
        redundant.append(pd.get_dummie(df[col]))
    titanic_redundant = pd.concat(redundant, axis=1)
    df = pd.concat((df,titanic_redundant),axis=1)
    df = df.drop(['Pclass','Sex','Embarked'],axis=1)
    df['Age'] = df['Age'].interpolate()
    X = df.values
    X = np.delete(X,1,axis=1)
    return X