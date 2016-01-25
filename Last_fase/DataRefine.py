import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import cross_validation


def get_manipulated_train():
    df = pd.read_csv('../data/train.csv',header=0)
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
    return (X,Y)



def generate_train_data():
    # train_frame = pd.read_csv('../data/train.csv',header=0)
    # train_frame=train_frame.drop('Survived',axis=1)
    # test_frame=pd.read_csv('../data/test.csv',header=0)
    # frames=[train_frame,test_frame]
    # df=pd.concat(frames,ignore_index=True)
    df = pd.read_csv('../data/train.csv',header=0)
    columns = ['Name','Ticket','Cabin']
    df = df.drop(columns,axis=1)
    redundant = []
    columns = ['Pclass','Sex','Embarked']
    for col in columns:
        redundant.append(pd.get_dummies(df[col]))
    titanic_redundant = pd.concat(redundant, axis=1)
    df = pd.concat((df,titanic_redundant),axis=1)
    df = df.drop(['Pclass','Sex','Embarked'],axis=1)
    df = df.dropna()
    Y = df['Age'].astype(int).values
    X = df.values
    X = np.delete(X,4,axis=1)
    return (X,Y)

def generate_refined_data():
    X,Y=generate_train_data()
    # return X[0]
    rf = ensemble.RandomForestClassifier(n_estimators=100)
    rf.fit(X,Y)
    df = pd.read_csv('../data/train.csv',header=0)
    columns = ['Name','Ticket','Cabin']
    df = df.drop(columns,axis=1)
    redundant = []
    columns = ['Pclass','Sex','Embarked']
    for col in columns:
        redundant.append(pd.get_dummies(df[col]))
    titanic_redundant = pd.concat(redundant, axis=1)
    df = pd.concat((df,titanic_redundant),axis=1)
    df = df.drop(['Pclass','Sex','Embarked'],axis=1)
    for i in range(len(df)):
        if pd.isnull(df.loc[i,'Age']):
            temp=df.loc[i].drop(['Age'])
            # print(temp.values.reshape(1,-1))
            age=rf.predict(temp.values.reshape(1,-1))
            # print(age[0])
            df.loc[i,'Age']=age[0]
    df_results = pd.DataFrame(df.astype('int'))
    df_results.to_csv('../data/refined_data.csv',index=False)
    Y = df['Survived'].values
    X = df.values
    X = np.delete(X,1,axis=1)
    return  cross_validation.train_test_split(X,Y,test_size=0.3,random_state=0)

def get_refined_data():
    df= pd.read_csv('../data/refined_data.csv',header=0)
    Y = df['Survived'].values
    X = df.values
    X = np.delete(X,1,axis=1)
    return  cross_validation.train_test_split(X,Y,test_size=0.3,random_state=0)


def get_test_data():
    df= pd.read_csv('../data/refined_data.csv',header=0)
    Y_train = df['Survived'].values
    X_train = df.values
    X_train = np.delete(X_train,1,axis=1)
    df = pd.read_csv('../data/test.csv',header=0)
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
    df['Fare'] = df['Fare'].interpolate()
    X_test = df.values
    return X_train,Y_train,X_test

if __name__=="__main__":
    generate_refined_data()
    pass