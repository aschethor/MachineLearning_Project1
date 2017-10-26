import numpy as np
import utils
from proj1_helpers import create_csv_submission
#def prj1_load_data():
#    A=np.genfromtxt('train.csv',delimiter=',',dtype=None)
#    X=np.asarray(A[1:,2:],dtype=float)
#    Y=np.zeros([X.shape[0],1])
#    for i in range(X.shape[0]):
#        if A[i+1][1]=='b':
#            Y[i][0]=1
#    return A,X,Y
#
#A,X,Y = prj1_load_data()

def generate_tx(X):
    tx = np.ones([X.shape[0],1])
    tx = np.c_[tx,X]
    return tx

a=np.genfromtxt('./train.csv',delimiter=",")
gender = np.genfromtxt('train.csv',dtype=None, delimiter=",",skip_header=1, usecols=[1],converters={0: lambda x: 0 if 'b' in x else 1})

Y=np.where(gender==b's',1,0)
X=a[1:,2:]

xt=np.where(X==-999,0,X)
mean=np.mean(xt,axis=0)
X=np.where(X==-999,mean,X)
std=np.std(X,axis=0)
X=(X-mean)/std
X=utils.sigmoid(X)
X=utils.build_poly(X,3)
tx=generate_tx(X)

loss,w = utils.least_squares(Y[:200000],tx[:200000,:])

#loss2,w2 = utils.sig_ls(Y[:200000],tx[:200000,:],np.ones(31)/31,10000,0.1,20)
#w1,loss1=utils.logistic_regression_stoch(Y[:100],tx[:100,:],np.zeros(31),100,0.1,1)


def accuracy(Y,tx,w,sig=False):
    p=np.dot(tx,w)
    if sig:
        p=utils.sigmoid(p)
    N=len(Y)
    pred=np.where(p>0.5,1,0)
    acc=np.sum(Y==pred)/N
    return acc

def predict(tx,w):
    p=np.dot(tx,w)
    pred=np.where(p>0.5,1,-1)
    return pred
    
    
print(accuracy(Y[np.where(Y==1)],tx[np.where(Y==1),:],w))
print(accuracy(Y[:200000],tx[:200000,:],w))
print(accuracy(Y[200000:],tx[200000:,:],w))
print(loss)
print(utils.compute_loss(Y[200000:],tx[200000:,:],w))
#print(accuracy(Y[:200000],tx[:200000,:],w2,sig=True))
#print(accuracy(Y[200000:],tx[200000:,:],w2,sig=True)) 


#test=np.genfromtxt('./test.csv',delimiter=",",skip_header=1)
print(1)
Xtest=test[:,2:]
Xtest=np.where(Xtest==-999,mean,Xtest)
Xtest=(Xtest-mean)/std
Xtest=utils.sigmoid(Xtest)
Xtest=utils.build_poly(Xtest,3)
Xtest=generate_tx(Xtest)
pred=predict(Xtest,w)

#for k in range(5):
#    Xtest=test[k*100000:(k+1)*100000,2:]
#    Xtest=np.where(Xtest==-999,mean,Xtest)
#    Xtest=(Xtest-mean)/std
#    Xtest=utils.sigmoid(Xtest)
#    Xtest=utils.build_poly(Xtest,10)
#    Xtest=generate_tx(Xtest)
#    pred=np.concatenate((predict(Xtest,w),pred))

create_csv_submission(test[:,0],pred,'prediction2.csv')
