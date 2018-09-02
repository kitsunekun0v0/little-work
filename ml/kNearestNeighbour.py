import numpy as np
from past.builtins import xrange
import math

def predict(Tr, X, k=1):
	"""
	Tr: training data. mxd numpy matrix.
	X: testing data. nxD numpy matrix.
	k: number of nearest neighbour.
	Return: predict label y in shape (n,1).
	"""
	dist = calculate_distance(Tr,X)
	n = dist.shape[0]
    y_pred = np.zeros(n)
    for i in xrange(n):
    	idx = np.argsort(dists[i])[0:k]
        closest_y = self.y_train[idx]
        (label, count) = np.unique(closest_y,return_counts=True)
        idx = np.argmax(count)
        y_pred[i] = label[idx]

    return y_pred

def calculate_distance(Tr, X):
	"""
	Tr: training data
	X: testing data.
	Return: distance between training data and testing data
	"""
	mult = np.matmul(X, Tr.T)
	test_sqr_sum = np.sum(X*X, axis=1)
	train_sqr_sum = np.sum(Tr.T*Tr.T, axis=0)
	outer_add = test_sqr_sum[:,np.newaxis] + train_sqr_sum #do outer operation of two arrays
    print(outer_add.shape)
    dists = np.sqrt(outer_add - 2*mult)

    return dist

