import numpy as np

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
    ret = np.ones(len(x))
    for i in range(1,degree+1):
        ret=np.c_[ret,np.power(x,i)]
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

