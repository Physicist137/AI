import numpy as np

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
				self.activation = self.sigmoid
				self.derivative = self.drsigmoid
			else:
				raise IncorrectConfiguration("Invalid activation")
			
		elif callable(activation)  and  callable(derivative):
			self.activation = activation
			self.derivative = derivative

		else:
			raise IncorrectConfiguration("Invalid Layer configuration")
	

	def __repr__(self):
		return "Layer(units={})".format(self.units)

		
	# Sigmoid activation.
	def sigmoid(self, z): return 1.0 / (1.0 + np.exp(-z))
	def drsigmoid(self, z): return z*(1-z)

	# Tanh activation.
	def tanh(self, z): return np.tanh(z)
	def drtanh(self, z): return 1 - z*z;
	


class NeuralNetwork:
	def __init__(self, features):
		self.init = 0.01
		self.features features
		self.layers = []
		self.w = np.array([])
		self.b = np.array([])


	def add_layer(self, layer=None, activation=None, derivative=None, units=0):
		# Add the layers.
		if isinstance(layer, Layer): self.layers.append(layer)
		else: self.layers.append(Layer(activation, derivative, units))

		# Randomly initialize the parameters of the layer.
		if len(self.layers) == 1:
			b = np.zeros((self.layers[0].units, 1))
			w = self.init * np.random.randn(self.layers[0].units, features)
		else:
			b = np.zeros((self.layers[-1].units, 1))
			w = self.init * np.random.randn(
				self.layers[-1].units, self.layers[-2].units
			)
	
		# Append layer parameters.
		self.w = np.append(self.w, w)
		self.b = np.append(self.b, b)
	

