import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class DataGenerator:
	def __init__(self):
		self.distributions = []

	def add_multivariate_normal(self, mean=np.array([0,0]), covariance=None, label=0):
		# Make sure mean is a rank-1 tensor.
		assert len(mean.shape) == 1

		# Use identity covariance if there's none.
		if covariance == None: covariance = np.eye(mean.size)

		# Save the distribution.
		self.distributions.append({
			'type' : 'multivariate_normal',
			'mean' : mean,
			'covariance' : covariance,
			'label' : label
		})
	
	
	def find_index_of_label(self, label):
		indices = []
		for i in range(0, len(self.distributions)):
			if self.distributions[i]['label'] == label: indices.append(i)

		if len(indices) != 0: return indices
		else: return None
	

	def generate_from_distribution(self, index):
		if self.distributions[index]['type'] == 'multivariate_normal':
			return np.random.multivariate_normal(
				self.distributions[index]['mean'],
				self.distributions[index]['covariance']
			)


	def generate_from_label(self, label=None):
		if isinstance(label, type(self.distributions[0]['label'])):
			indices = self.find_index_of_label(label)
			if indices is None: return None

			rand = np.random.randint(0, len(indices))
			index = indices[rand]
			return self.generate_from_distribution(index)

		else:
			index = np.random.randint(0, len(self.distributions))
			return self.generate_from_distribution(index)
			
		
	def generate_data(self, amount=10):
		each = int(np.ceil(amount / len(self.distributions)))
		points = []
		labels = []

		for distribution in self.distributions:
			if distribution['type'] == 'multivariate_normal':
				points.append(np.random.multivariate_normal(
					distribution['mean'],
					distribution['covariance'],
					each
				))
				labels += [distribution['label']] * each

		return np.concatenate(points).T, np.array(labels).T


