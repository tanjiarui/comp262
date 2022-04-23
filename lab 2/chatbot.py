import json, numpy as np, tensorflow as tf, matplotlib.pyplot as plt, shelve
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

json_file = json.load(open('intents.json'))
sentences, intents, responses = list(), list(), dict()


def convert_data(item):
	for sentence in item['patterns']:
		sentences.append(sentence)
		intents.append(item['tag'])
	for response in item['responses']:
		if item['tag'] not in responses.keys():
			responses[item['tag']] = [response]
		else:
			responses[item['tag']].append(response)


# data preprocessing
MAX_SEQUENCE_LENGTH = 20
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 16
[convert_data(i) for i in json_file['intents']]
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)  # converting text to a vector of word indexes
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))
# initial padding of 0s, until vector is of size MAX_SEQUENCE_LENGTH
sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(intents)
x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=.2)


# modeling
def build_model():
	input_layer = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,))
	embedding = tf.keras.layers.Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_layer)
	layer = tf.keras.layers.GlobalAveragePooling1D()(embedding)
	layer = tf.keras.layers.Dense(16, activation='relu')(layer)
	layer = tf.keras.layers.Dense(16, activation='relu')(layer)
	output_layer = tf.keras.layers.Dense(label_encoder.classes_.size, activation='softmax')(layer)
	ann = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	return ann


# plot metric
def metric(history):
	epoch = train_history.epoch
	train_accuracy = train_history.history['accuracy']
	val_accuracy = train_history.history['val_accuracy']
	plt.title('train vs validation')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.legend()
	plt.plot(epoch, train_accuracy, color='b', label='train')
	plt.plot(epoch, val_accuracy, color='r', label='validation')
	plt.legend()
	plt.show()
	# validation
	prediction = [np.argmax(predict) for predict in model.predict(x_test)]
	print(classification_report(y_test, prediction))


model = build_model()
model.summary()
train_history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=300)
metric(train_history)
'''
			precision    recall   f1-score    support

	0           0.00      0.00      0.00         2
	1           1.00      1.00      1.00         1
	2           0.00      0.00      0.00         0
	4           0.00      0.00      0.00         2
	5           0.00      0.00      0.00         0
	6           1.00      1.00      1.00         1
	7           0.00      0.00      0.00         1

accuracy                            0.29         7
macro avg       0.29      0.29      0.29         7
weighted avg    0.29      0.29      0.29         7
'''
train_history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=500)
metric(train_history)
# save encoder and model
model.save('model')
encoder = shelve.open('encoder')
encoder['tokenizer'] = tokenizer
encoder['label_encoder'] = label_encoder
encoder['responses'] = responses
encoder.close()