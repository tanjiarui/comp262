import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report

data = pd.read_csv('dataset')
analyzer = SentimentIntensityAnalyzer()

def score(text):
	compound = analyzer.polarity_scores(text).pop('compound')
	if compound >= .05:
		return 'positive'
	elif -.05 < compound < .05:
		return 'neutral'
	else:
		return 'negative'

data['prediction'] = (data['summary'] + ' ' + data['reviewText']).apply(lambda text: score(text))
print(classification_report(data['sentiment'], data['prediction']))
'''
			precision    recall  f1-score   support

negative        0.14      0.31      0.19        35
neutral         0.00      0.00      0.00        57
positive        0.92      0.92      0.92       908

accuracy                            0.85      1000
macro avg       0.35      0.41      0.37      1000
weighted avg    0.84      0.85      0.84      1000
'''
'''
thoughts to include in the report:
1. why does the requirement require to split dataset and train the model? there is no training since it's a rule-based model
2. the metric is totally valueless. the dataset is extremely unbalanced. A higher metric can be achieved by predicting all as 'positive'
'''