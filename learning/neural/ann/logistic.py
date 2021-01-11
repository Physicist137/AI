import scipy.io
import numpy as np

class WrongLabel(Exception): pass

# Implements Logistic Regression.
class Logistic:
	def __init__(self, w=None, b=None, data=None, labels=None, rate=0.01):
		# Training data
		self.data = data
		self.labels = labels

		# Parameters:
		if self.data is not None  and   w is None:
			self.w = np.zeros(self.data.shape[0])
			self.b = 0
		else:
			self.w = w
			self.b = b

		# Hyperparameters
		self.learning_rate = rate


	def loadmat(self, filename):
		data = scipy.io.loadmat(filename)
		self.w = data['w']
		self.b = data['b']
	
	def savemat(self, filename, do_compression=False):
		data = {'w' : w, 'b' : b}
		scipy.io.savemat(filename, data, do_compression=do_compression)
	
	def saveall(self, filename, do_compression=False):
		data = {
			'parametes' : {
				'w' : self.w,
				'b' : self.b
			},
			'training' : {
				'data' : self.data,
				'labels' : self.labels
			}
		}
		scipy.io.savemat(filename, data, do_compression=do_compression)
	
	def loadall(self, filename):
		x = scipy.io.loadmat(filename)
		self.w = x['parameters']['x']
		self.b = x['parameters']['b']
		self.data = x['training']['data']
		self.labels = x['training']['labels']

	def sigmoid(self, z):
		return 1.0 / (1 + np.exp(-z))
	
	def forward_propagation(self, feature):
		if self.w is None: return None
		if self.b is None: return None
		return self.sigmoid(np.dot(feature, self.w) + self.b)
	
	def backward_propagation(self, feature, labels):
		a = self.forward_propagation(feature, labels)
		dz = a - labels
		dw = np.dot(feature, dz)
		db = np.sum(dz)
		return (dw, db)
	
	def loss_function(self, feature, labels):
		prediction = self.forward_propagation(feature)
		if labels == 0: return np.log(prediction)
		elif labels == 1: return np.log(1-prediction)
		else: raise WrongLabel()
	
	def cost_function(self):
		a = self.forward_data().reshape(self.labels.size)
		yes = np.dot(self.labels, np.log(a))
		no = np.dot(1-self.labels, np.log(1-a))
		return -yes-no
	
	def forward_data(self):
		ww = self.w.reshape(1, self.w.size)
		z = np.dot(ww, self.data) + self.b
		return self.sigmoid(z.reshape(z.size))
	
	def backward_data(self):
		a = self.forward_data().reshape(1, self.labels.size)
		y = self.labels.reshape(1, self.labels.size)
		dz = a - y
		dw = np.dot(self.data, dz.T)
		db = np.sum(dz)

		return (
			dw.reshape(dw.size), 
			db
		)
	
	def training_iterate(self, num=1):
		for i in range(0, num):
			dw, db = self.backward_data()
			self.w -= self.learning_rate * dw
			self.b -= self.learning_rate * db


