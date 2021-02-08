import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)



# Add noise to the images.
noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)



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
	epochs=15,
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
	ax = plt.subplot(2, n, i + 1)
	plt.title("original + noise")
	plt.imshow(tf.squeeze(x_test_noisy[i]))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	# display reconstruction
	bx = plt.subplot(2, n, i + n + 1)
	plt.title("reconstructed")
	plt.imshow(tf.squeeze(decoded_imgs[i]))
	plt.gray()
	bx.get_xaxis().set_visible(False)
	bx.get_yaxis().set_visible(False)


plt.savefig('denoiser.png')

