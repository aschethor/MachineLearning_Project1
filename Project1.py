import numpy as np
import utils

def prj1_load_data():
    print("loading data...")
    A=np.genfromtxt('train.csv',delimiter=',',dtype=None)
    print("creating X,Y...")
    X=np.asarray(A[1:,2:],dtype=float)
    Y=np.zeros([X.shape[0],1])
    for i in range(X.shape[0]):
        if A[i+1][1]=='b':
            Y[i][0]=1
    print("done.")
    return A,X,Y

#A,X,Y = prj1_load_data()
X = np.load("Xtrain.npy")
Y = np.load("Ytrain.npy")

def generate_tx(X):
    tx = np.ones([X.shape[0],1])
    tx = np.c_[tx,X]
    return tx

#don't use this function for the whole dataset!!! ;)
def generate_tx_cross(X):
    tx = np.ones([X.shape[0],1])
    tx = np.c_[tx,X]
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            tx = np.c_[tx,X[:,i]*X[:,j]]
    return tx

def train_least_squares(tx,Y):
    size = tx.shape[0]
    batch_size = 1000
    final_w = np.zeros(tx.shape[1])
    for i in range(size/batch_size):
        loss,w = utils.least_squares(Y[(i*batch_size):((i+1)*batch_size)],tx[(i*batch_size):((i+1)*batch_size),:])
        final_w += w
    return final_w*batch_size/size

def predict(tx,w):
    return np.sign(np.dot(tx,w)-0.5)*0.5+0.5

def accuracy(tx,Y,w):
    return float(np.sum(Y == predict(tx,w)))/tx.shape[0]

tx=generate_tx(X)

print("calculating least squares")

loss,w = utils.least_squares(Y[:10000],tx[:10000,:])
Ypred = np.round(np.dot(tx,w))
accuracy = float(np.sum(Y==Ypred))/250000

print("calculating logistic regression")

w1,loss1=utils.logistic_regression(Y[:500],tx[:500,:],np.zeros([tx.shape[1],1]),1000,0.1)
Ypred2 = np.dot(tx,w1)
Ypred2 = (np.sign(Ypred2-0.5)+1)/2
accuracy2 = float(np.sum(Y==Ypred2))/250000
blubb=0