import numpy as np, tensorflow as tf, shelve
from keras.preprocessing.sequence import pad_sequences
from random import choice

# load model and encoder
model = tf.keras.models.load_model('model')
encoder = shelve.open('encoder')
tokenizer = encoder['tokenizer']
label_encoder = encoder['label_encoder']
responses = encoder['responses']
encoder.close()

print('how can I help you?')
while True:
	sentence = [input()]
	if sentence[0] == 'exit':
		break
	sentence = tokenizer.texts_to_sequences(sentence)
	sentence = pad_sequences(sentence, maxlen=20)
	intent = np.argmax(model.predict(sentence))
	intent = label_encoder.inverse_transform([intent])[0]
	print(choice(responses[intent]))