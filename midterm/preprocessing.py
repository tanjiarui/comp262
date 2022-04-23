import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

documents = ['The girl tricks a cat_', 'Cat tricks the girl!', 'girl eats food.', '<the cat beats fish>']


def handle_string(text):
	result = list()
	for token in text.lower().split():
		if token not in stopwords.words('english'):
			result.append(token)
	return ' '.join(result)


for index in range(len(documents)):
	documents[index] = handle_string(documents[index])
tfidf = TfidfVectorizer()
terry_tfidf = tfidf.fit_transform(documents)
print('IDF for all words in the vocabulary', tfidf.idf_)
print('all words in the vocabulary', tfidf.get_feature_names_out())
print('TFIDF representation for all documents in our corpus\n', terry_tfidf.toarray())