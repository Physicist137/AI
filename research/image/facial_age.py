import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config

# Set up a few parameters
batch_size = 32
image_height = 200
image_width = 200
seed = 3483
# 389, 2624, 2725, 2884
# stopped search at 7750.

# Get the images.
data_dir = pathlib.Path(dataset_path)
images_path = list(data_dir.glob('*/*.png'))



def crop_image_masks(image, to_crop=(15,50), value=np.array([1.0, 0, 1.0])):
	image_size_x = 200
	image_size_y = 200
	
	to_crop = np.random.randint(to_crop[0], to_crop[1])
	center_x = np.random.randint(to_crop, image_size_x-to_crop)
	center_y = np.random.randint(to_crop, image_size_y-to_crop)
	
	shape = (image_size_x, image_size_y, 3)
	zeros = np.zeros(shape)
	ones = np.ones(shape)
	
	x1 = center_x-to_crop
	x2 = center_x+to_crop
	y1 = center_y-to_crop
	y2 = center_y+to_crop
	
	zeros[x1:x2, y1:y2] = value
	ones[x1:x2, y1:y2] = np.zeros(3)
	
	return zeros, ones


def crop_image_with_masks(image, to_crop=(15,50), value=np.array([1.0, 0, 1.0])):
	zeros, ones = crop_image_masks(image, to_crop, value)
	tmask = tf.convert_to_tensor(zeros, dtype=tf.float32)
	tfilter = tf.convert_to_tensor(ones, dtype=tf.float32)
	return image * tfilter + tmask


def normalize_image(filename):
	image = tf.io.read_file(filename)
	image = tf.image.decode_png(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	return image


def denormalize_image(image):
	return tf.convert_to_tensor(tf.cast(tf.round(image * 255.0), tf.int32), dtype=tf.int32)


def crop_image(filename):
	return crop_image_with_masks(normalize_image(filename))


def parse_image(filename):
	image = tf.io.read_file(filename)
	image = tf.image.decode_png(image)
	image = tf.image.convert_image_dtype(image, tf.float32)
	
	zeros, ones = crop_image_masks(image)
	tmask = tf.convert_to_tensor(zeros, dtype=tf.float32)
	tfilter = tf.convert_to_tensor(ones, dtype=tf.float32)
	cropped = image * tfilter + tmask
	
	return cropped, image


def show(images, filename='show.png'):
	img = 0
	
	if not isinstance(images, list):
		images = [images]
	
	for image in images:
		if isinstance(image, list):
			for picture in image:
				img += 1
				plt.subplot(len(images), len(image), img)
				plt.xticks([])
				plt.yticks([])
				plt.grid(False)
				plt.imshow(picture, cmap=plt.cm.binary)
		
		else:
			img += 1
			plt.subplot(1, len(images), img)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(image, cmap=plt.cm.binary)
	
	plt.savefig(filename)


class Autoencoder(tf.keras.models.Model):
	def __init__(self):
		super(Autoencoder, self).__init__()
		
		# Encoding network.
		# Perhaps to reduce more: 200x200x3 is too larger than 25x25x8
		self.encoder = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(200, 200, 3)),
			tf.keras.layers.Conv2D(32, (10,10), activation='relu', padding='same', strides=2),
			tf.keras.layers.Conv2D(8, (8,8), activation='relu', padding='same', strides=2),
			tf.keras.layers.Conv2D(2, (4,4), activation='relu', padding='same', strides=2),
		])
		
		# Decoding network. 13x13x4.
		self.decoder = tf.keras.Sequential([
			tf.keras.layers.Conv2DTranspose(2, (4,4), strides=2, activation='relu', padding='same'),
			tf.keras.layers.Conv2DTranspose(8, (8,8), strides=2, activation='relu', padding='same'),
			tf.keras.layers.Conv2DTranspose(32, (10,10), strides=2, activation='relu', padding='same'),
			tf.keras.layers.Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same'),
		])
		
		
	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded



class SavePictureCallback(tf.keras.callbacks.Callback):
	def __init__(self, image, cropped, initial_epoch=0):
		super(SavePictureCallback, self).__init__()
		self.initial_epoch = initial_epoch
		self.image = image
		self.cropped = cropped
		self.pictures = []
	
	def on_epoch_end(self, epoch, logs=None):
		name = 'epoch_pictures3/epoch{:03d}.png'.format(epoch+1+self.initial_epoch)
		picture = denormalize_image(tf.squeeze(autoencoder(tf.expand_dims(cropped, axis=0)))).numpy().astype("uint8")
		self.pictures.append(picture)
		
		show([
			denormalize_image(image).numpy().astype("uint8"),
			denormalize_image(cropped).numpy().astype("uint8"),
			picture
		], filename = name)


# Checkpoint callback.
checkpoint_path = "model_checkpoint3/cp-{epoch:03d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_path, 
	verbose=1, 
	save_weights_only=True,
)


# Get the image datasets.
list_dataset = tf.data.Dataset.list_files(dataset_path + '/*/*.png', seed=seed)
image_dataset = list_dataset.map(normalize_image)
noise_dataset = list_dataset.map(crop_image)
pair_dataset = list_dataset.map(parse_image)
batch_dataset = pair_dataset.batch(128)

for cropped, image in pair_dataset: break

picture_callback = SavePictureCallback(image, cropped, initial_epoch=30)

show([denormalize_image(image).numpy().astype("uint8"), denormalize_image(cropped).numpy().astype("uint8")])


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
autoencoder.load_weights(checkpoint_path.format(epoch=0))
autoencoder.fit(batch_dataset, epochs=300, callbacks=[cp_callback, picture_callback])

