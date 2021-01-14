import numpy as np

class Activation:
	# Sigmoid Activation function.
	@staticmethod
	def sigmoid(z):
		return np.where(z >= 0,
			1.0 / (1 + np.exp(-z)),
			np.exp(z)  / (1.0 + np.exp(z))
		)
	
	@staticmethod
	def da_sigmoid(a):
		return a*(1-a)
	
	
	# Tanh Activation Function.
	@staticmethod
	def tanh(z): return np.tanh(z)
	@staticmethod
	def da_tanh(a): return 1 - a*a


	# Rectified Linear Unit
	@staticmethod
	def relu(z): return np.where(z >= 0, z, 0.0)
	@staticmethod
	def da_relu(z): return np.where(z >= 0, 1.0, 0.0)


	# Leaky Rectified Linear Unit
	@staticmethod
	def leaky_relu(z): return np.where(z >= 0, z, 0.01*z)
	@staticmethod
	def da_leaky_relu(z): return np.where(z >= 0, 1.0, 0.01)


# Useful links.
# http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
# https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
# https://numpy.org/devdocs/reference/generated/numpy.where.html

