import json, numpy as np, nltk
from keras.preprocessing.text import Tokenizer
from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
nltk.download('averaged_perceptron_tagger')

train_file = json.load(open('train_PlayMusic_full.json', encoding='iso-8859-2'))
test_file = json.load(open('validate_PlayMusic.json', encoding='iso-8859-2'))
print(type(train_file))
print(len(train_file))
train_data = [i['data'] for i in train_file['PlayMusic']]
test_data = [i['data'] for i in test_file['PlayMusic']]


def convert_data(datalist):
	output = list()
	for data in datalist:
		s, p = list(), list()
		for phrase in data:
			words = phrase['text'].strip().split(" ")
			while '' in words:
				words.remove('')
			if 'entity' in phrase.keys():
				label = phrase['entity']
				labels = [label + '-{}'.format(i + 1) for i in range(len(words))]
			else:
				labels = ['O'] * len(words)
			s.extend(words)
			p.extend(labels)
		output.append([s, p])
	return output


train_data = convert_data(train_data)
test_data = convert_data(test_data)
train_texts = [' '.join(i[0]) for i in train_data]
test_texts = [' '.join(i[0]) for i in test_data]
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3

print('preparing embedding matrix.')
# first, build index mapping words in the embeddings set to their embedding vector
embeddings_index = dict()
with open('glove.6B.100d.txt', encoding='utf-8') as file:
	for line in file:
		values = line.split()
		word = values[0]
		coefficients = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefficients
print('found %s word vectors in Glove embeddings.' % len(embeddings_index))


def get_embeddings(word):
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is None:
		# words not found in embedding index will be all-zeros.
		embedding_vector = np.zeros(shape=(EMBEDDING_DIM,))
	return embedding_vector


tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)  # Converting text to a vector of word indexes
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
print('found %s unique tokens.' % len(word_index))

"""
get features for all words in the sentence
features:
- word context: a window of 2 words on either side of the current word, and current word.
- POS context: a window of 2 POS tags on either side of the current word, and current tag. 
input: sentence as a list of tokens.
output: list of dictionaries. each dict represents features for that word.
"""


def sent2feats(sentence):
	feats = []
	sen_tags = pos_tag(sentence)  # This format is specific to this POS tagger!
	for i in range(0, len(sentence)):
		word = sentence[i]
		wordfeats = {'word': word}
		# word features: word, prev 2 words, next 2 words in the sentence.
		if i == 0:
			wordfeats['prevWord'] = wordfeats['prevSecondWord'] = '<S>'
		elif i == 1:
			wordfeats['prevWord'] = sentence[0]
			wordfeats['prevSecondWord'] = '</S>'
		else:
			wordfeats['prevWord'] = sentence[i - 1]
			wordfeats['prevSecondWord'] = sentence[i - 2]
		# next two words as features
		if i == len(sentence) - 2:
			wordfeats['nextWord'] = sentence[i + 1]
			wordfeats['nextNextWord'] = '</S>'
		elif i == len(sentence) - 1:
			wordfeats['nextWord'] = '</S>'
			wordfeats['nextNextWord'] = '</S>'
		else:
			wordfeats['nextWord'] = sentence[i + 1]
			wordfeats['nextNextWord'] = sentence[i + 2]

		# POS tag features: current tag, previous and next 2 tags.
		wordfeats['tag'] = sen_tags[i][1]
		if i == 0:
			wordfeats['prevTag'] = wordfeats['prevSecondTag'] = '<S>'
		elif i == 1:
			wordfeats['prevTag'] = sen_tags[0][1]
			wordfeats['prevSecondTag'] = '</S>'
		else:
			wordfeats['prevTag'] = sen_tags[i - 1][1]

			wordfeats['prevSecondTag'] = sen_tags[i - 2][1]
		# next two words as features
		if i == len(sentence) - 2:
			wordfeats['nextTag'] = sen_tags[i + 1][1]
			wordfeats['nextNextTag'] = '</S>'
		elif i == len(sentence) - 1:
			wordfeats['nextTag'] = '</S>'
			wordfeats['nextNextTag'] = '</S>'
		else:
			wordfeats['nextTag'] = sen_tags[i + 1][1]
			wordfeats['nextNextTag'] = sen_tags[i + 2][1]

		# Adding word vectors
		vector = get_embeddings(word)
		for iv, value in enumerate(vector):
			wordfeats['v{}'.format(iv)] = value

		feats.append(wordfeats)
	return feats


# extract features from the conll data, after loading it.
def get_feats_conll(conll_data):
	f, l = list(), list()
	for sentence in conll_data:
		f.append(sent2feats(sentence[0]))
		l.append(sentence[1])
	return f, l


# crf model
x_train, y_train = get_feats_conll(train_data)
x_test, y_test = get_feats_conll(test_data)
crf = CRF(algorithm='lbfgs', c1=0.1, c2=10, max_iterations=50)  # all_possible_states=True)
crf.fit(x_train, y_train)
labels = list(crf.classes_)
# test
y_pred = crf.predict(x_test)
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
binarizer = MultiLabelBinarizer()
y_test = binarizer.fit_transform(y_test)
y_pred = binarizer.transform(y_pred)
print(classification_report(y_test, y_pred))