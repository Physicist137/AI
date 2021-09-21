import bisect
import numpy as np


# Some random corpus.
corpus = "Yeah, I think that, at some point we should have something considerably better here, presumably, yeah. That is, hopefully."


# Process Text Corpus.
class ProcessCorpus:
	def __init__(self, corpus, isolated=None):
		# Get the isolated characters.
		if isinstance(isolated, str):
			self.isolated = list(isolated)
		elif isinstance(isolated, list):
			self.isolated = isolated
		elif isolated is None:
			self.isolated = ['.', ',', ':', ' ', '?', '!', "'s", '-']
		else:
			raise ValueError('isolated is not a list.')
		
		
		# Get the corpus and create vocabulary.
		self.words = corpus.lower().split(' ')
		self.create_vocabulary()
		

	def create_vocabulary(self):
		self.vocabulary = []
		for word in self.words:
			was_appended = False
			for token in self.isolated:
				if token in word:
					item = word.replace(token, '')
					if item not in self.vocabulary:
						self.vocabulary.append(item)
						was_appended = True
						break
			
			if was_appended == False:
				if word not in self.vocabulary:
					self.vocabulary.append(word)
		
		# Sort the vocabulary.
		self.vocabulary.sort()


	def vocabulary_size(self):
		return len(self.vocabulary)



class SkipGram:
	# Initialize.
	# dimensions: The dimension of the encoding.
	# window: The size of the window taking during training.
	def __init__(self, corpus, dimensions=10, window=4):
		# Get corpus.
		self.processed_corpus = ProcessCorpus(corpus)
		
		# Randomly initialize matrix encoding.
		self.encoding_center_matrix = np.random.rand(self.processed_corpus.vocabulary_size(), dimensions) / 1e3
		self.encoding_context_matrix = np.random.rand(self.processed_corpus.vocabulary_size(), dimensions) / 1e3
		
		# Window initialization.
		self.window = window


	# Get the vector encoding of a given word.
	def vector_encoding(self, word, type=None):
		# Make use of the assumption the vocabulary must be sorted already.
		# https://stackoverflow.com/questions/3196610/searching-a-sorted-list
		# https://docs.python.org/3/library/bisect.html
		
		if type == None:
			type = 'center'
		
		if type in ['center']:
			index = bisect.bisect_left(self.processed_corpus.vocabulary, word)
			if index != len(self.processed_corpus.vocabulary)  and  self.processed_corpus.vocabulary[index] == word:
				return self.encoding_center_matrix[index]
			else:
				raise ValueError('Word is not in vocabulary.')
		
		elif type in ['context']:
			index = bisect.bisect_left(self.processed_corpus.vocabulary, word)
			if index != len(self.processed_corpus.vocabulary)  and  self.processed_corpus.vocabulary[index] == word:
				return self.encoding_context_matrix[index]
			else:
				raise ValueError('Word is not in vocabulary.')
		
		else:
			raise ValueError('Invalid type.')



	# Get the probability of center word due to context word.
	# Softmax function implementation.
	def probability_from_string(self, center, context):
		center_encoding = self.vector_encoding(center)
		context_encoding = self.vector_encoding(context)
		product = np.dot(center_encoding, context_encoding)
		total = np.dot(self.encoding_context_matrix, center_encoding)
		num = np.exp(product - np.max(total))
		den = np.exp(total - np.max(total))
		return num / np.sum(den)


	def probability_from_encoding(self, center, context):
		product = np.dot(center, context)
		total = np.dot(self.encoding_context_matrix, center.T)
		num = np.exp(product - np.max(total))
		den = np.exp(total - np.max(total))
		return num / np.sum(den)
	
