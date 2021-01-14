import scipy.io
import numpy as np
from learning.neural.ann.activation import Activation

class IncorrectConfiguration(Exception): pass


class Layer:
	def __init__(self, activation=None, dractivation=None, units=0):
		# Get the number of units
		if units <= 0  or isinstance(units, int) == False:
			raise IncorrectConfiguration("Number of units is invalid")
		else:
			self.units = units
		

		# Get the activation function.
		if activation is None:
			raise IncorrectConfiguration("An activation needs to be chosen")
		
		elif isinstance(activation, str):
			if activation == 'sigmoid':
				self.activation = Activation.sigmoid
				self.derivative = Activation.da_sigmoid
				self.function = 'sigmoid'
			elif activation == 'tanh':
				self.activation = Activation.tanh
				self.derivative = Activation.da_tanh
				self.function = 'tanh'
			elif activation == 'relu':
				self.activation = Activation.relu
				self.derivative = Activation.da_relu
				self.function = 'relu'
			elif activation == 'leaky_relu':
				self.activation = Activation.leaky_relu
				self.derivative = Activation.da_leaky_relu
				self.function = 'leaky_relu'
			elif activation == 'identity'  or  activation == 'none':
				self.activation = Activation.identity
				self.derivative = Activation.da_identity
				self.function = 'identity'
			else:
				raise IncorrectConfiguration("Invalid activation")
			
		elif callable(activation)  and  callable(derivative):
			self.activation = activation
			self.derivative = derivative
			self.function = None

		else:
			raise IncorrectConfiguration("Invalid Layer configuration")
	

	def __repr__(self):
		return "Layer(units={})".format(self.units)

		
class NeuralNetwork:
	def __init__(self, features, initialization=0.01, learning_rate=0.01):
		# Hyperparameters
		self.init = initialization
		self.learning_rate = learning_rate

		# Counting of parameters, units, and input features.
		self.parameters = 0
		self.units = 0
		self.features = features

		# Learning parameters
		self.layers = []
		self.w = []
		self.b = []


	def save_model(self, filename, do_compression=False):
		activations = []
		for layer in self.layers: 
			if layer.function is None:
				activations.append('none')
				print("WARNING: You have custom activations that weren't saved")
				print("When loading, you will have to add them manually")
			else: activations.append(layer.function)

		data = {
			'activations' : activations,
			'w' : self.w,
			'b' : self.b
		}

		scipy.io.savemat(filename, data, do_compression=do_compression)


	def load_model(self, filename):
		data = scipy.io.loadmat(filename)
		self.w = data['w']
		self.b = data['b']
		activations = data['activations']
		self.features = self.w[0].shape[1]
		self.parameters = 0

		L = len(self.b)
		for l in range(0, L):
			self.layers.append(Layer(
				activation=activations[l],
				units=self.b[l].size
			))

			self.parameters += self.w[l].size + self.b[l].size



	def add_layer(self, layer=None, activation=None, derivative=None, units=0):
		# Add the layers.
		if isinstance(layer, Layer): self.layers.append(layer)
		else: self.layers.append(Layer(activation, derivative, units))

		# Randomly initialize the parameters of the layer.
		if len(self.layers) == 1:
			self.units += self.layers[0].units
			self.parameters += self.layers[0].units * (self.features + 1)
			b = np.zeros((self.layers[0].units, 1), dtype=np.float64)
			w = self.init * np.random.randn(
				self.layers[0].units, self.features
			).astype('float64')
		else:
			self.units += self.layers[0].units
			self.parameters += self.layers[-1].units * (self.layers[-2].units + 1)
			b = np.zeros((self.layers[-1].units, 1), dtype=np.float64)
			w = self.init * np.random.randn(
				self.layers[-1].units, self.layers[-2].units
			).astype('float64')
	
		# Append layer parameters.
		self.w.append(w)
		self.b.append(b)
	

	def forward_propagation(self, feature, acache=False):
		if acache == False:
			a = feature
			for l in range(0, len(self.layers)):
				z = np.dot(self.w[l], a) + self.b[l]
				a = self.layers[l].activation(z)
	
			return a
	
		if acache == True:
			cache = []
			a = feature
			for l in range(0, len(self.layers)):
				z = np.dot(self.w[l], a) + self.b[l]
				a = self.layers[l].activation(z)
				cache.append(a)

			cache.append(feature)
			return a, cache
	

	def backward_propagation(self, label, acache):
		# This assumes last unit is sigmoid. FIX this.
		#das = label / acache[L-1] + (1-label) / (1-acache[L-1])
		#da = np.sum(das, axis=0, keepdims=True)
		L = len(self.layers)
		dz = acache[L-1] - label

		if len(label.shape) == 1: m = 1
		else: m = label.shape[1]

		wcache = [0] * len(self.w)
		bcache = [0] * len(self.b)
		for l in reversed(range(0, len(self.layers))):
			if (l != L-1): dz = da * self.layers[l].derivative(acache[l])
			dw = np.dot(dz, acache[l-1].T) / m
			db = np.sum(dz, axis=1, keepdims=True) / m
			da = np.dot(self.w[l].T, dz)

			wcache[l] = dw
			bcache[l] = db

		return wcache, bcache
	

	def train_iteration(self, feature, label, num=1):
		for k in range(0, num):
			a, acache = self.forward_propagation(feature, acache=True)
			dw, db = self.backward_propagation(label, acache)

			for l in range(0, len(self.w)):
				self.w[l] -= self.learning_rate * dw[l]
				self.b[l] -= self.learning_rate * db[l]

