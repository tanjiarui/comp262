import os, re, string, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_json('meta digital music.json', dtype=str, lines=True)
drop_columns = ['category', 'tech1', 'fit', 'tech2', 'feature', 'main_cat', 'similar_item', 'date', 'imageURL', 'imageURLHighRes', 'details']
for column in data.columns:
	print('distinct value of %s' % column)
	print(data[column].unique())
data[data.isin(['[]', '', 'nan', '{}'])] = None
print(data.isnull().sum())
'''
category           74340
tech1              74347
description        37821
fit                74347
title                677
also_buy           64960
tech2              74347
brand              14471
feature            74258
rank                5938
also_view          60758
main_cat               0
similar_item       74347
date               74342
price              27897
asin                   0
imageURL           51124
imageURLHighRes    51124
details             1540
'''
data.drop(columns=drop_columns, inplace=True)


def string_handle(text):
	crap = re.compile(r'RT|@[^\s]*|[^\s]*…|\n+|\t+')
	text = crap.sub(r'', text)
	url = re.compile(r'https?://\S+|www\.\S+')
	text = url.sub(r'', text)  # remove urls
	html = re.compile(r'<.*?>')
	text = html.sub(r'', text)  # remove html tags
	emoji_pattern = re.compile(
		'['
		'\U0001F1E0-\U0001F1FF'  # flags (iOS)
		'\U0001F300-\U0001F5FF'  # symbols & pictographs
		'\U0001F600-\U0001F64F'  # emoticons
		'\U0001F680-\U0001F6FF'  # transport & map symbols
		'\U0001F700-\U0001F77F'  # alchemical symbols
		'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
		'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
		'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
		'\U0001FA00-\U0001FA6F'  # Chess Symbols
		'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
		'\U00002702-\U000027B0'  # Dingbats
		'\U000024C2-\U0001F251'
		']+'
	)
	text = emoji_pattern.sub(r'', text)  # remove emojis
	table = str.maketrans('', '', string.punctuation)
	text = text.translate(table)  # remove punctuations
	return text if text != '' else None


# text clean
data['description'] = data['description'].apply(lambda x: string_handle(x.lower()) if x else None)
data['title'] = data['title'].apply(lambda x: string_handle(x.lower()) if x else None)
data['brand'] = data['brand'].apply(lambda x: string_handle(x.lower()) if x else None)
data = data.drop(data[data['title'].isnull()].index).drop_duplicates(subset=['title']).reset_index(drop=True)
data['title'].to_csv('title', index_label='index')  # index map
# vectorize textual columns
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['title'])
start = 0
partition = 1000
if not os.path.exists('cosine similarity'):
	os.mkdir('cosine similarity')
# memory overflow. must split the matrix as several partitions
while partition < tfidf_matrix.shape[0]:
	cosine_sim = pd.DataFrame(linear_kernel(tfidf_matrix[start:partition], tfidf_matrix[start:partition]), index=list(range(start, partition)), columns=list(range(start, partition)))
	path = os.path.join('cosine similarity', 'matrix-' + str(start) + '-' + str(partition))
	cosine_sim.to_csv(path, index_label='index')
	start = partition
	partition += 1000
partition = tfidf_matrix.shape[0]
cosine_sim = pd.DataFrame(linear_kernel(tfidf_matrix[start:partition], tfidf_matrix[start:partition]), index=list(range(start, partition)), columns=list(range(start, partition)))
path = os.path.join('cosine similarity', 'matrix-' + str(start) + '-' + str(partition))
cosine_sim.to_csv(path, index_label='index')