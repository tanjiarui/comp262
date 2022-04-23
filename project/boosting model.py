import pandas as pd, nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
nltk.download('stopwords')

data = pd.read_csv('dataset')

def remove_stop(text):
	result = list()
	for token in text.split(' '):
		if token not in stopwords.words('english'):
			result.append(token)
	return ' '.join(result)

data['text'] = (data['summary'] + ' ' + data['reviewText']).apply(lambda text: remove_stop(text))
vectorizer = CountVectorizer()
tf_idf = TfidfTransformer().fit_transform(vectorizer.fit_transform(data['text']))
weight = tf_idf.toarray()

data['sentiment'].replace(['positive', 'neutral', 'negative'], [0, 1, 2], inplace=True)
x_train, x_test, y_train, y_test = train_test_split(weight, data['sentiment'], test_size=.3)
model = XGBClassifier(use_label_encoder=False)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
prediction = model.predict(x_test)
print(classification_report(y_test, prediction))
'''
			precision    recall   f1-score   support

	0           0.91      1.00      0.95       273
	1           0.00      0.00      0.00        18
	2           0.00      0.00      0.00         9

accuracy                            0.91       300
macro avg       0.30      0.33      0.32       300
weighted avg    0.83      0.91      0.87       300
'''