import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class DataGenerator:
	def __init__(self):
		self.distributions = []
		self.amount = 0

	def add_multivariate_normal(self, mean=np.array([0,0]), covariance=None, label=0):
		
		# Make sure mean is linear.
		assert len(mean.shape) == 1

		# Use unit covariance
		if covariance == None: covariance = np.eye(mean.size)

		# Save the distribution.
		self.distributions.append({
			'type' : 'multivariate_normal',
			'mean' : mean,
			'covariance' : covariance,
			'label' : label
		})

		self.amount += 1
	
	def generate_data(self, amount=10):
		each = int(np.ceil(amount / self.amount))
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

		return np.concatenate(points), np.array(labels).T
	

	@staticmethod
	def visualize_data_2D(data, labels):
		x = data.T[0]
		y = data.T[1]
		colors = ['blue', 'orange']

		fig = plt.figure(figsize=(8,8))
		plt.scatter(
			x, y, 
			c=labels, 
			cmap=matplotlib.colors.ListedColormap(colors)
		)

		plt.show()
	

