import tensorflow as tf


def load_audio(file_path):
	binary = tf.io.read_file(file_path)
	return tf.audio.decode_wav(binary)


def write_audio(file_path, audio, sample_rate):
	encode = tf.audio.encode_wav(audio, sample_rate)
	tf.io.write_file(file_path, encode)


def get_stft(audio, frame_length=512, frame_step=128, milisecond_length=None, channels='complex'):
	if milisecond_length is not None:
		length = tf.cast(tf.math.ceil(sample_rate.numpy() * ms_window / 1e3), dtype=tf.int32)
	else:
		length = frame_length
	
	
	audio_channels = tf.transpose(audio)
	transform = []
	for i in range(audio_channels.shape[0]):
		result = tf.signal.stft(audio_channels[i], frame_length=length, frame_step=frame_step)

		if channels == 'complex': transform.append(result)
		elif channels == 'realimag': transform.append([tf.math.real(result), tf.math.imag(result)])
		elif channels == 'absarg': transform.append([tf.math.abs(result), tf.math.angle(result)])
		elif channels == 'complete': transform.append([tf.math.real(result), tf.math.imag(result), tf.math.abs(result), tf.math.angle(result)])
		else: transform.append(result)
	
	
	if not channels == 'complex':
		return tf.transpose(tf.convert_to_tensor(transform), [0, 2, 3, 1])
	else:
		return tf.convert_to_tensor(transform)



def get_inverse_stft(transformed, frame_length=512, frame_step=128, milisecond_length=None):
	if milisecond_length is not None:
		length = tf.cast(tf.math.ceil(sample_rate.numpy() * ms_window / 1e3), dtype=tf.int32)
	else:
		length = frame_length

	result = []
	for i in range(transformed.shape[0]):
		inverse = tf.signal.inverse_stft(transformed[i], frame_length=length, frame_step=frame_step)
		result.append(inverse)

	return tf.transpose(tf.convert_to_tensor(result))


