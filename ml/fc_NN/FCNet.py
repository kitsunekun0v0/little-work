import numpy as np
from layer import *

class FCNet(object):
	"""
	a fully connected network with user defined number of hidden layers.
	activate function is ReLU for each hidden layer.
	"""
	def __int__(self, inputs, outputs, hidden, sd, r):
		"""
		initialise a network.

		input:
		inputs: number of neuron in input layer
		output: number of neuron in output layer (= number of classes)
		hidden: list of integer indicates the number of neuron in each layer
		sd: standard deviation of normal distribution weights initialised from
		r: l2 regularisation strength
		"""
		self.num_layer = len(hidden)+1
		self.r = r
		self.params = {}

		# initialise weights and biases
		for i in range(self.num_layer):
        	if i==0:
        		self.params['W%d' % (i+1)] = np.random.normal(0,sd,inputs*hidden[i]).reshape((inputs,hidden[i]))
        		self.params['b%d' % (i+1)] = np.zeros(hidden[i])
        	elif i==(self.num_layer-1):
        		self.params['W%d' % (i+1)] = np.random.normal(0,sd,outputs*hidden[i-1]).reshape((hidden[i-1],outputs))
        		self.params['b%d' % (i+1)] = np.zeros(outputs)
        	else:
        		self.params['W%d' % (i+1)] = np.random.normal(0,sd,hidden[i-1]*hidden[i]).reshape((hidden[i-1],hidden[i]))
        		self.params['b%d' % (i+1)] = np.zeros(hidden[i])

    def backprop(self, X, y=None):
    	"""
    	calculate loss for the network and find gradient of loss with respect to weights and biases

    	input:
    	X: input data. shape is (N, d1, d2, ...)
    	y: labels. shaoe is (N,), or no input y value.

    	return:
    	loss: loss of the network
    	grad: a dictionary contain gradient with respect to each weight and bias
    	score: retrun classification score if y is not provided.
    	"""
    	# implement the forward pass 
    	a, cache = {}, {}
        a['a1'], cache['cache1'] = affine_activate_forward(X, self.params['W1'], self.params['b1'])
        for i in range(1,self.num_layer-1):
        	a['a%d' % (i+1)], cache['cache%d' % (i+1)] = affine_activate_forward(a['a%d' % (i)], self.params['W%d' % (i+1)],self.params['b%d' % (i+1)])
        a['a%d' % (self.num_layer)], cache['cache%d' % (self.num_layer)] = affine_forward(a['a%d' % (self.num_layer-1)], self.params['W%d' % (self.num_layer)], self.params['b%d' % (self.num_layer)])

        score = a['a%d' % (self.num_layer)]

        if y is None:
        	return score

        # implement backward pass
        loss, dx = loss_softmax(score, y)

        # add regularisation to loss
        regu = 0
        for i  in range(self.num_layer):
        	regu +=(np.sum(self.params['W%d' % (i+1)]*self.params['W%d' % (i+1)]))
        regu *= (0.5*self.r)
        loss +=regu

        # calculate gradient
        grad = {}
        n = self.num_layer
        da, grad['W%d'%(n)], grad['b%d'%(n)] = affine_backward(dx, cache['cache%d'%(n)])
        grad['W%d'%(n)] += self.r*self.params['W%d'%(n)]
        n -= 1
        while n>0:
            da, grad['W%d'%(n)], grad['b%d'%(n)] = affine_activate_backward(da, cache['cache%d'%(n)])
            grad['W%d'%(n)] += self.r*self.params['W%d'%(n)]
            n -= 1
        
        return loss, grad 




