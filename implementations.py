import numpy as np
from random import randint
from helpers import batch_iter

def compute_loss(y, tx, w):
    """Calculate the loss using mse."""
    N = len(y)
    return np.sum(np.square(y-np.dot(tx,w)))/2/N

def least_squares(y, tx):
    """calculate the least squares solution."""
    U,s,V = np.linalg.svd(np.dot(tx.transpose(),tx));
    tx_inv = np.dot(V.transpose(),np.dot(np.diag(np.reciprocal(s)),U.transpose()))
    weights = np.dot(np.dot(tx_inv,tx.transpose()),y)
    return compute_loss(y,tx,weights),weights

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    ret = x
    for i in range(2,degree+1):
        ret=np.concatenate((ret,x**i),axis=1)
    return ret

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:int(np.floor(ratio*x.shape[0]))], indices[int(np.floor(ratio*x.shape[0]+1)):]
    return x[training_idx],y[training_idx],x[test_idx],y[test_idx]

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    U,s,V = np.linalg.svd(np.dot(tx.transpose(),tx));
    tx_inv = np.dot(V.transpose(),np.dot(np.diag(np.reciprocal(s+lambda_)),U.transpose()))
    weights = np.dot(np.dot(tx_inv,tx.transpose()),y)
    return compute_loss(y,tx,weights),weights

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    return np.dot(tx.transpose(),np.dot(tx,w)-y)/N

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w)
        gradient = compute_gradient(y,tx,w)
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N = len(y)
    return np.dot(tx.transpose(),np.dot(tx,w)-y)/N


from random import randint

def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    N = len(y)
    batch_size = 10
    for n_iter in range(max_iters):
        i = [randint(0,N) for p in range(batch_size)]
        loss = compute_loss(y[i],tx[i],w)
        gradient = compute_gradient(y[i],tx[i],w)
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    M=np.dot(tx,w)
    loss=np.log(1+np.exp(M))-y*M
    return np.sum(loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    M=np.dot(tx,w)
    grad=np.dot(tx.T,sigmoid(M)-y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss=calculate_loss(y,tx,w)
    grad=calculate_gradient(y,tx,w)
    w=w-gamma*grad
    return loss, w 


def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    M=np.dot(tx,w)
    S=np.diag((sigmoid(M)*(1-sigmoid(M))).ravel())
    H=np.dot(tx.T,np.dot(S,tx))
    return H

def logistic_regression_step(y, tx, w):
    """return the loss, gradient, and hessian."""
    M=np.dot(tx,w)
    loss=np.sum(np.log(1+np.exp(M))-y*M)
    grad=np.dot(tx.T,sigmoid(M)-y)
    S=np.diag((sigmoid(M)*(1-sigmoid(M))).ravel())
    H=np.dot(tx.T,np.dot(S,tx))
    return loss,grad,H

def logistic_regression(y,tx,initial_w,max_iters,gamma):
    """ Implement logistic regresssion algorithm """
    w=initial_w
    L=[]
    for n in range(max_iters):
        loss,w = learning_by_newton_method(y,tx,w,gamma)
        L.append(loss)
    return w,L

def learning_by_newton_method(y, tx, w,gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss,grad,H=logistic_regression_step(y,tx,w)
    delta=np.linalg.solve(-H,grad)
    w+=delta*gamma
    return loss, w

    
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss=calculate_loss(y,tx,w)+lambda_/2*np.dot(w.T,w)
    grad=calculate_gradient(y,tx,w)+lambda_*w
    return loss,grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss,grad=penalized_logistic_regression(y,tx,w,lambda_)
    w-=gamma*grad
    return loss, w


def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """ Implement the regularized logistic regression algorithm """
    w=initial_w
    for n in range(max_iters):
        loss,w = learning_by_penalized_gradient(y,tx,w,gamma,lambda_)
    return w,loss
    

def logistic_regression_stoch(y,tx,initial_w,max_iters,gamma,batch_size):
    """ Implement sthochastic logistic regression algorithm """
    batch=batch_iter(y,tx,batch_size,max_iters)
    w=initial_w
    for (yb,txb) in batch:
        loss,w=learning_by_gradient_descent(yb,txb,w,gamma)
    return loss,w



def predict(tx,w,b=-1):
    """ predict the classes of the elements """
    p=np.dot(tx,w)
    pred=np.where(p>0.5,1,b)
    return pred

