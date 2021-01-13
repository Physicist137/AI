import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Visualize a two input network.
class VisualizeTwoInputNeuralNetwork:
	def __init__(self):
		self.ind = None
		self.input_datasweep = None
		self.output_datasweep = None

	# vhttps://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
	def set_input_datasweep(self, domain=None, max_size=100000, to_return=False):
		# Get the domain.
		if isinstance(domain, (list, tuple)):
			if isinstance(domain[0], (list, tuple)):
				xmin = domain[0][0]
				xmax = domain[0][1]
				ymin = domain[1][0]
				ymax = domain[1][1]
			elif isinstance(domain[0], (int,float)):
				xmin = domain[0]
				xmax = domain[1]
				ymin = domain[0]
				ymax = domain[1]
			else:
				raise ValueError("Couldn't understand domain argument")
			
		else: 
			raise ValueError("Couldn't understand domain argument")
				

		# Generate the input data sweep at given domain.
		ind = int(np.floor(np.sqrt(max_size)))
		x = np.linspace(xmin,xmax,ind)
		y = np.linspace(ymin,ymax,ind)
		X,Y = np.meshgrid(x,y)
		XY = np.array([X.T.flatten(),Y.T.flatten()])

		self.ind = ind
		self.input_datasweep = XY
		if to_return == True: return self.input_datasweep


	def set_output_datasweep(self, sweep_out):
		self.output_datasweep = sweep_out.reshape(self.ind, self.ind)


	# https://matplotlib.org/3.3.3/tutorials/introductory/sample_plots.html
	# https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.pcolormesh.html#matplotlib.pyplot.pcolormesh
	def visualize(self, data, labels, visualize_network=False, visualize_data=True):
		fig = plt.figure(figsize=(8,8))
		if visualize_network == True:
			X = np.max(self.input_datasweep[0].reshape(self.ind, self.ind),axis=1)
			Y = self.input_datasweep[1].reshape(self.ind, self.ind)[0]
			C = self.output_datasweep
			plt.pcolormesh(X,Y,C,shading='auto')

		if visualize_data == True:
			x = data[0]
			y = data[1]
			colors = ['blue', 'orange']

			plt.scatter(
				x, y, 
				c=labels, 
				cmap=matplotlib.colors.ListedColormap(colors)
			)

		plt.show()



