import numpy as np
from past.builtins import xrange

def train(X, y, W, r, epoch, l, batch_size): 
	"""
	X: (n x d) traning data, n is the number of samples and d is the number fo feature
	y: label in shape (n,)
	W: initialised weight
	r: regularisation strength
	epoch: number of iteration
	l: learning rate
	batch_size: number of sample used for each iteration
	return: trained weight
	"""

	n, d = X.shape

	for i in xrange(epoch):
		idx = np.random.choice(n, batch_size, replace=False)
		X_batch = X[idx]
        y_batch = y[idx]
		grad = softmax_grad(X_batch,y_batch,W,r)
		W = W - l*grad

	return W


def predict(X, W): 
	"""
	X: (n x d) testing data, n is the number of samples and d is the number of features
	W: trained weight
	return: predicted class
	"""

	p = softmax(X,W)
	pre_y = np.argmax(p, axis=1)

    return pre_y


def softmax_grad(X, y, W,r):
	gradient = np.zeros_like(W)
    n_X, d_X = X.shape

    # probability of each class given x, p(y|x,w)
    p = softmax(X, W)
    mask = np.eye(p.shape[1])[y]
    # calculate the gradient
    gradient = np.dot(X.T,(p-mask))
    # add regularisation
    gradient = gradient/n_X + 2*r*W

    return gradient

def softmax(X,W):
	n_X, d_X = X.shape

	all_score = np.dot(X,W)
    all_score -= np.max(all_score, axis=1).reshape((n_X,1)) # for numeric stability

    # probability of each class given x, p(y|x,w)
    all_score = np.exp(all_score)
    p = all_score/np.sum(all_score, axis=1)[:,np.newaxis]

    return p

