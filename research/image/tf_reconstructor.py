import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)




# Instead of noise.. let's clip a couple images.
#window_crop = 0.2
#to_crop = int(round(28 * window_crop))


def crop_images(dataset, to_crop=5, value=1.0):
	centers = tf.random.uniform((dataset.shape[0],2), to_crop, 28-to_crop, dtype=tf.int32).numpy()
	copy = np.array(dataset, copy=True)
	for i in range(dataset.shape[0]):
		x1 = centers[i][0]-to_crop
		x2 = centers[i][0]+to_crop
		y1 = centers[i][1]-to_crop
		y2 = centers[i][1]+to_crop
		copy[i][x1:x2, y1:y2] = value
	
	return copy


x_train_noisy = crop_images(x_train)
x_test_noisy = crop_images(x_test)



# Define the denoiser autoencoder.
class Denoise(tf.keras.models.Model):
	def __init__(self):
		super(Denoise, self).__init__()
		self.encoder = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(28, 28, 1)), 
			tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
			tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2)
		])
		
		self.decoder = tf.keras.Sequential([
			tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
			tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
			tf.keras.layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')
		])
		
	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


# Compile the model.
autoencoder = Denoise()
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

# Train the model.
autoencoder.fit(
	x_train_noisy,
	x_train,
	epochs=20,
	shuffle=True,
	validation_data=(x_test_noisy, x_test)
)



encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# Show examples.
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
	# display original + noise
	ax = plt.subplot(3, n, i + 1)
	plt.title("original + noise")
	plt.imshow(tf.squeeze(x_test_noisy[i]))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	# display reconstruction
	bx = plt.subplot(3, n, i + n + 1)
	plt.title("reconstructed")
	plt.imshow(tf.squeeze(decoded_imgs[i]))
	plt.gray()
	bx.get_xaxis().set_visible(False)
	bx.get_yaxis().set_visible(False)

	# display original.
	bx = plt.subplot(3, n, i + 2*n + 1)
	plt.title("original")
	plt.imshow(tf.squeeze(x_test[i]))
	plt.gray()
	bx.get_xaxis().set_visible(False)
	bx.get_yaxis().set_visible(False)


plt.savefig('reconstructor.png')