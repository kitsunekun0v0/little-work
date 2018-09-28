import numpy as np

def affine_forward(x, w, b):
	"""
	computation of forward pass.

	input:
	x: input data which should be a numpy array of shape (N, d1, d2, ...)
	w: a numpy array of weight
	b: a numpy array of bias. (h,) h is the number of neurons in hidden layer.

	return:
	value: x.w+b
	cache: tuple (x,w,b)
	"""
	d = np.prod(x.shape[1:])
    x_reshaped = x.reshape(x.shape[0],d)
    value = np.dot(x_reshaped,w) + b[np.newaxis,:]
    cache = (x, w, b)

    return value, cache

def activate_forward(x):
	"""
	forward pass for an activation function, ReLU is used

	input:
	x: numpy array of any shape.

	return:
	value: output of activation function
	cache: x
	"""
	value = x
    value[x<0] = 0
    cache = x

    return value, cache

def affine_activate_forward(x, w, b):
	"""
	compute forward pass for affine and activation in one function
	input -> affine -> ReLU -> next layer
	"""
	y, cache_affine = affine_forward(x, w, b)
    value, cache_activate = activate_forward(y)
    cache = (cache_affine, cache_activate)
    return value, cache

def loss_softmax(x, y):
	"""
	compute loss and gradient

	input:
	x: score of each class. shape is (N, C), N is # of sample, C is # of classes
	y: corresponding class label. shape is (N,)

	return:
	loss: total loss for all samples
	grad: derivation of loss respect to x
	"""
	score = x - np.max(x, axis=1, keepdims=True)
	p = score/np.sum(score, axis=1)[:,np.newaxis]
	mask = np.eye(score.shape[1])[y] # select probability of correct class
    loss = np.sum(-np.log(np.sum(p*mask,axis=1)))/x.shape[0]
    grad = (p-mask)/x.shape[0]

    return loss, grad

def affine_backward(grad, cache):
	"""
	input:
	grad: derivation of upper layer
	cache: tuple of x, weights and biases of this layer

	return:
	dx: gradient with respect to x
	dw: gradient with respect to weights
	db: gradient with respect to biases
	"""
	x, w, b = cache
	db = np.sum(grad, axis=0).reshape(b.shape)
    d = np.prod(x.shape[1:])
    x_reshaped = x.reshape(x.shape[0],d)
    dw = np.dot(x_reshaped.T, grad)
    dx = np.dot(grad, w.T).reshape(x.shape)

    return dx, dw, db

def activate_backward(grad, cache):
	"""
	input:
	grad: derivation of upper layer
	cache: x of this layer

	return:
	dx: gradient with respect to x
	"""
	x = cache
	dx = grad.copy()
	dx[x<=0] *= 0.5 # avoid dead ReLU

	return dx

def affine_activate_backward(grad, cache):
	cache_affine, cache_activate = cache
	dx = activate_backward(grad, cache_activate)
	dx, dw, db = affine_backward(dx, cache_affine)

	return dx, dw, db



