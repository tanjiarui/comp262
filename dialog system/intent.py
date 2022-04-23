import numpy as np, pandas as pd, tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.initializers.initializers_v2 import Constant


# load dataset
def get_train_data(filename):
	with open(filename) as f:
		contents = f.read()
		s, l, i = list(), list(), list()
		for line in contents.strip().split('\n'):
			words, labs = [i.split(' ') for i in line.split('\t')]
			s.append(words[1:-1])
			l.append(labs[1:-1])
			i.append(labs[-1])
		return np.array(s, dtype=object), np.array(l, dtype=object), np.array(i, dtype=object)


sentences, labels, intents = get_train_data('atis-2.train.w-intent.iob')
train_sentences = [' '.join(i) for i in sentences]
train_texts = train_sentences
train_labels = intents.tolist()
# removing any intents with #
vals = list()
for i in range(len(train_labels)):
	if "#" in train_labels[i]:
		vals.append(i)
for i in vals[::-1]:
	train_labels.pop(i)
	train_texts.pop(i)

print("Number of training sentences :", len(train_texts))
print("Number of unique intents :", len(set(train_labels)))
print(type(train_texts))
print(len(train_labels))
print(len(vals))
print(len(vals))
print(len(sentences))


def get_test_data(filename):
	df = pd.read_csv(filename, delim_whitespace=True, names=['word', 'label'])
	beg_indices = list(df[df['word'] == 'BOS'].index) + [df.shape[0]]
	s, l, i = list(), list(), list()
	for index in range(len(beg_indices[:-1])):
		s.append(df[beg_indices[index] + 1:beg_indices[index + 1] - 1]['word'].values)
		l.append(df[beg_indices[index] + 1:beg_indices[index + 1] - 1]['label'].values)
		i.append(df.loc[beg_indices[index + 1] - 1]['label'])
	return np.array(s, dtype=object), np.array(l, dtype=object), np.array(i, dtype=object)


sentences, labels, intents = get_test_data('atis-2.dev.w-intent.iob')
test_sentences = [' '.join(i) for i in sentences]
test_texts = test_sentences
test_labels = intents.tolist()
new_labels = set(test_labels) - set(train_labels)
# removing any intents with #
vals = []
for i in range(len(test_labels)):
	if "#" in test_labels[i]:
		vals.append(i)
	elif test_labels[i] in new_labels:
		print(test_labels[i])
		vals.append(i)
for i in vals[::-1]:
	test_labels.pop(i)
	test_texts.pop(i)
print("Number of testing sentences :", len(test_texts))
print("Number of unique intents :", len(set(test_labels)))

# data preprocessing
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)  # converting text to a vector of word indexes
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
test_labels = encoder.transform(test_labels)
# converting this to sequences to be fed into neural network. Max seq. len is 1000 as set earlier
# initial padding of 0s, until vector is of size MAX_SEQUENCE_LENGTH
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
# split the training data into a training set and a validation set
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_labels = train_labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])
x_train = train_data[:-num_validation_samples]
y_train = train_labels[:-num_validation_samples]
x_val = train_data[-num_validation_samples:]
y_val = train_labels[-num_validation_samples:]
# this is the data we will use for CNN and RNN training
print('splitting the train data into train and valid is done')

# modeling
embeddings_index = dict()
with open('glove.6B.100d.txt', encoding='utf-8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefficients = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefficients

print('found %s word vectors in Glove embeddings.' % len(embeddings_index))
# prepare embedding matrix - rows are the words from word_index, columns are the embeddings of that word from glove.
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
	if i > MAX_NUM_WORDS:
		continue
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in embedding index will be all-zeros
		embedding_matrix[i] = embedding_vector

# load these pre-trained word embeddings into an Embedding layer
# note that we set trainable is false to keep the embeddings fixed
embedding_layer = tf.keras.layers.Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), input_length=MAX_SEQUENCE_LENGTH, trainable=False)
print('preparing of embedding matrix is done')
model = tf.keras.Sequential()
model.add(embedding_layer)
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(5))
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(5))
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(train_labels.max() + 1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# train the model. tune to validation set
model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_val, y_val))
# evaluate on test set:
_, accuracy = model.evaluate(test_data, test_labels)
print('test accuracy with CNN:', accuracy)
# train a CNN model with an embedding layer which is being trained on the fly instead of using the pre-trained embeddings
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(MAX_NUM_WORDS, 128))
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(5))
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(5))
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(train_labels.max() + 1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_val, y_val))
_, accuracy = model.evaluate(test_data, test_labels)
print('test accuracy with CNN:', accuracy)