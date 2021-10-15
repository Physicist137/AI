# Based on: https://www.tensorflow.org/text/tutorials/text_generation



# Load tensorflow libraries.
# IF you are running this, perhaps you might want to delete this.
# https://github.com/tensorflow/tensorflow/issues/48868#issuecomment-841396124
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/cuda/bin")

# Load tensorflow.
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

# Load other stuff.
import numpy as np
import time




# Hyperparameters.
seq_length = 100
embedding_dim = 256
rnn_units = 1024

# Some other parameters.
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 5



# Download and load the text. Define its vocabulary.
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))


# Text Vectorization.
# Create the preprocessing layers which can convert chars and IDs.
ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
def text_from_ids(ids):
	return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


# Define the dataset in terms of IDs.
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


# Define batches from the dataset. Sequences of characters are given as batch from a given size.

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


# Arrange input-->target sequences.
def split_input_target(sequence):
	input_text = sequence[:-1]
	target_text = sequence[1:]
	return input_text, target_text


# Define the actual dataset: input-->output .
dataset = sequences.map(split_input_target)
dataset = (
	dataset
	.shuffle(BUFFER_SIZE)
	.batch(BATCH_SIZE, drop_remainder=True)
	.prefetch(tf.data.experimental.AUTOTUNE)
)


class LanguageModelGRU(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, rnn_units):
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
		self.dense = tf.keras.layers.Dense(vocab_size)
	
	
	def call(self, inputs, states=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x, training=training)
		
		if states is None:
			states = self.gru.get_initial_state(x)
		
		x, states = self.gru(x, initial_state=states, training=training)
		x = self.dense(x, training=training)
		
		if return_state:
			return x, states
		else:
			return x


# Define the model.
model = LanguageModelGRU(
	vocab_size=len(ids_from_chars.get_vocabulary()),
	embedding_dim=embedding_dim,
	rnn_units=rnn_units
)


class OneStep(tf.keras.Model):
	def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
		super().__init__()
		self.temperature = temperature
		self.model = model
		self.chars_from_ids = chars_from_ids
		self.ids_from_chars = ids_from_chars

		# Create a mask to prevent "[UNK]" from being generated.
		skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
		sparse_mask = tf.SparseTensor(
			values=[-float('inf')]*len(skip_ids),
			indices=skip_ids,
			dense_shape=[len(ids_from_chars.get_vocabulary())]
		)
		
		self.prediction_mask = tf.sparse.to_dense(sparse_mask)

	@tf.function
	def generate_one_step(self, inputs, states=None):
		# Convert strings to token IDs.
		input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
		input_ids = self.ids_from_chars(input_chars).to_tensor()

		# Run the model.
		# predicted_logits.shape is [batch, char, next_char_logits]
		predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
		
		# Only use the last prediction.
		predicted_logits = predicted_logits[:, -1, :]
		predicted_logits = predicted_logits/self.temperature
		
		# Apply the prediction mask: prevent "[UNK]" from being generated.
		predicted_logits = predicted_logits + self.prediction_mask

		# Sample the output logits to generate token IDs.
		predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
		predicted_ids = tf.squeeze(predicted_ids, axis=-1)

		# Convert from token ids to characters
		predicted_chars = self.chars_from_ids(predicted_ids)

		# Return the characters and model state.
		return predicted_chars, states



one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
def generate_text(initial_text, amount_letters=10):
	start = time.time()
	states = None
	next_char = tf.constant([initial_text])
	result = [next_char]

	for n in range(amount_letters):
		next_char, states = one_step_model.generate_one_step(next_char, states=states)
		result.append(next_char)

	result = tf.strings.join(result)
	end = time.time()
	print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
	print('\nRun time:', end - start)




# Build the model by calling the model at least once (or by usind model.build()).
for input_example_batch, target_example_batch in dataset.take(1):
	example_batch_predictions = model(input_example_batch)
	#print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")



# model.summary()

# Define loss function for the model.
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile the model with ADAM optimization.
model.compile(optimizer='adam', loss=loss)

# Train model and test it.
# model.fit(dataset, epochs=60)
# generate_text("Hello", 500)




# Configure model checkpoints, for saving it during epochs.
# Directory where the checkpoints will be saved
# Name of the checkpoint files
#checkpoint_dir = './training_checkpoints'
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
#checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#	filepath=checkpoint_prefix,
#	save_weights_only=True
#)


#one_step_model = OneStep(model, chars_from_ids, ids_from_chars)


# Train the model.
# history = model.fit(dataset, epochs=30)