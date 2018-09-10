import numpy as np
from past.builtins import xrange

def softmax_cross_entropy(X, y, W, r):
	"""
	A vectorised version softmax function which can work fast as it contains no loops.
  The loss function is cross-entropy error function.

	inputs:
	X: (n x d) training data. n is the numaber of samples, d is the number of features.
	y: label in shape (n,)
	W: initial weight in shape (d x c). c is the number of classes. 
	r: regularisation strength

	return gradient of W and the total loss for optimisation and tuning.
	"""
  
  gradient = np.zeros_like(W)
  n_X, d_X = X.shape

  all_score = np.dot(X,W)
  all_score -= np.max(all_score, axis=1).reshape((n_X,1)) # for numeric stability

  # probability of each class given x, p(y|x,w)
  all_score = np.exp(all_score)
  p = all_score/np.sum(all_score, axis=1)[:,np.newaxis]

  # calculate the loss
  mask = np.eye(all_score.shape[1])[y] # select probability of correct class
  loss = np.sum(-np.log(np.sum(p*mask,axis=1)))

  # add regularisation
  regu = r*(np.sum(W*W))
  loss = loss/n_X + regu

  # calculate the gradient
  gradient = np.dot(X.T,(p-mask))
  # add regularisation
  gradient = gradient/n_X + 2*r*W


  """
  loop version for more clear understanding.

  for i in xrange(n_X):
    all_score = np.dot(W.T,X[i])
    all_score -= np.max(all_score) # for mumeric stability

    # calculate the loss 
    true_class_score = all_score[y[i]]
    norm_s = np.sum(np.exp(all_score))
    p_i = np.exp(true_class_score)/norm_s
    i_loss = (-np.log(p_i))
    loss += i_loss

    # calculate the gradient
    for j in xrange(W.shape[1]):
      if j==y[i]:
        gradient[:,j] += (p_i-1)*X[i]
      else:
        p_j = np.exp(all_score[j])/norm_s
        gradient[:,j] += p_j*X[i]

  # add regularisation into loss
  regu = r*(np.sum(np.diag(np.dot(W,W.T))))
  loss = loss/n_X + regu

  # add regularisation into gradient
  dW = gradient/n_X + 2*r*W 
  """


  return gradient, loss