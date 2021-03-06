# -*- coding: utf-8 -*-
"""exercise 2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Yh7MCpSjXBvegTItfxuyuEHGPMdqHid8
"""

import pandas as pd, re, string, random, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
nltk.download('punkt')
nltk.download('stopwords')

terry_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/artificial intelligence mini.csv').drop(columns='user')
terry_df.loc[0, 'text'] = terry_df.loc[0, 'text'][15:122].replace('sm', 's.m').lower()
terry_df.loc[1, 'text'] = terry_df.loc[1, 'text'][15:112].lower()
terry_df.loc[2, 'text'] = terry_df.loc[2, 'text'][20:138].replace('#', '').lower()
terry_df.loc[3, 'text'] = terry_df.loc[3, 'text'][17:116].replace('#', '').lower()

model = KeyedVectors.load_word2vec_format('/content/drive/MyDrive/Colab Notebooks/GoogleNews-vectors-negative300.bin.gz', binary=True)

for index, item in zip(terry_df.index, terry_df['text']):
	# removing digits
	text = re.sub(r'\d+', '', item)
	# removing punctuations
	text = text.translate(str.maketrans('', '', string.punctuation))
	tokens = word_tokenize(text)
	tokens = [token for token in tokens if token not in stopwords.words('english')]
	while True:
		try:
			choose_index = random.sample(range(0, len(tokens)), 2)
			synonyms = [model.most_similar(tokens[choose_index[0]])[0][0], model.most_similar(tokens[choose_index[1]])[0][0]]
			tokens[choose_index[0]], tokens[choose_index[1]] = synonyms
			break
		except KeyError:
			pass
	terry_df.loc[index, 'tweet'] = ' '.join(tokens)

file = open('/content/drive/MyDrive/Colab Notebooks/terry df', 'w')
for tweet in terry_df['tweet']:
	file.write(tweet + '\n')
file.close()