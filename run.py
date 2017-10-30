import numpy as np
from implementations import *
from proj1_helpers import create_csv_submission
import matplotlib.pyplot as plt
from random import randint

def generate_tx(X):
    """
    Do the preprocessing of the database: normalize the data, compute cross-terms,
    apply the sigmoid and construct a polynomial basis, 
    add a matrix of boolean representing the -999 values
    """
    ab=X==-999
    xt=np.where(X==-999,0,X)
    mean=np.mean(xt,axis=0)
    X=np.where(X==-999,mean,X)
    std=np.std(X,axis=0)
    X=(X-mean)/std
    Xs=sigmoid(X)
    Xsp=build_poly(Xs,7)
    txc=generate_tx_cross(X)
    tx=np.concatenate((Xsp,ab,txc),axis=1)
    return tx,mean,std

def generate_tx_cross(X):
    """
    Compute the cross-terms of the feature
    """
    tx = np.ones((X.shape[0],1))
    for i in range(X.shape[1]):
        tx = np.concatenate((tx,X[:,i].reshape((X.shape[0],1))*X),axis=1)
    return tx

# load the data
features=np.genfromtxt('./train.csv',delimiter=",")
label = np.genfromtxt('train.csv',dtype=None, delimiter=",",skip_header=1, usecols=[1],converters={0: lambda x: 0 if 'b' in x else 1})
Y=np.where(label==b's',1,0)
X=features[1:,2:]

#we just use the first 80000 values because of memory issues
Y=Y[:80000]
X=X[:80000,:]
tx,mean,std=generate_tx(X)

# we use least_squares to learn the model
loss,w = least_squares(Y[:75000],tx[:75000,:])

#predict the classes for the test dataset. 
#the for-loop is used because of memory issues
test=np.genfromtxt('./test.csv',delimiter=",",skip_header=1)
print(1)
Xtest=test[550000:,2:]
abtest=Xtest==-999
Xtest=np.where(Xtest==-999,mean,Xtest)
Xtest=(Xtest-mean)/std
Xstest=sigmoid(Xtest)
Xstest=build_poly(Xstest,7)
Xtest=generate_tx_cross(Xtest)
Xtest=np.concatenate((Xstest,abtest,Xtest),axis=1)
pred=predict(Xtest,w)

for k in range(10,-1,-1):
    print(1)
    Xtest=test[k*50000:(k+1)*50000,2:]
    abtest=Xtest==-999
    Xtest=np.where(Xtest==-999,mean,Xtest)
    Xtest=(Xtest-mean)/std
    Xstest=sigmoid(Xtest)
    Xstest=build_poly(Xstest,7)
    Xtest=generate_tx_cross(Xtest)
    Xtest=np.concatenate((Xstest,abtest,Xtest),axis=1)
    pred0=predict(Xtest,w)
    pred=np.concatenate((pred0,pred),axis=0)
    
create_csv_submission(test[:,0],pred,'prediction.csv')
