import numpy as np
import utils

def prj1_load_data():
    A=np.genfromtxt('train.csv',delimiter=',',dtype=None)
    X=np.asarray(A[1:,2:],dtype=float)
    Y=np.zeros([X.shape[0],1])
    for i in range(X.shape[0]):
        if A[i+1][1]=='b':
            Y[i][0]=1
    return A,X,Y

A,X,Y = prj1_load_data()

def generate_tx(X):
    tx = np.ones([X.shape[0],1])
    tx = np.c_[tx,X]
    return tx


tx=generate_tx(X)

loss,w = utils.least_squares(Y[:100],tx[:100,:])

w1,loss1=utils.logistic_regression(Y[:100],tx[:100,:],np.zeros([tx.shape[1],1]),1000,0.1)