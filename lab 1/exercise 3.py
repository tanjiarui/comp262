import pandas as pd, json, re, string, nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
nltk.download('punkt')

data = pd.read_csv('artificial intelligence data.csv').drop(columns='user')
positive = pd.read_csv('positive words', sep='\t', header=None)
negative = pd.read_csv('negative words', sep='\t', header=None)
positive.columns = ['words']
negative.columns = ['words']
pos_set = set(list(positive['words']))
neg_set = set(list(negative['words']))
with open('abbreviations.json', 'r', encoding='utf8')as fp:
	abbreviations = json.load(fp)

def convert_abbrev(word):
	return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
def string_handle(text):
	crap = re.compile(r'RT|@[^\s]*|[^\s]*\…')
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
	tokens = word_tokenize(text)  # word segmentation
	tokens = [convert_abbrev(word) for word in tokens]  # stretch abbr
	text = ' '.join(tokens)
	return text

data['tweet'] = data['text'].apply(lambda x:string_handle(x))
data.info()
'''
RangeIndex: 100 entries, 0 to 99
Data columns (total 3 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   sentiment  100 non-null    object
 1   text       100 non-null    object
 2   tweet      100 non-null    object
dtypes: object(3)
memory usage: 2.5+ KB
'''
data['tweet len'] = data['tweet'].str.count(' ') + 1

def assign_tag(tweet):
	text = set(tweet.split())
	length = len(text)
	pos_words = text & pos_set
	neg_words = text & neg_set
	pos_len = len(pos_words)
	neg_len = len(neg_words)
	pos_percent = pos_len / length if pos_len > 0 else 0
	neg_percent = neg_len / length if neg_len > 0 else 0
	if pos_percent > 0 or neg_percent > 0:
		sentiment = 'positive' if pos_percent > neg_percent else 'negative'
	else:
		sentiment = 'neutral'
	return pos_percent, neg_percent, sentiment

data['postitive'], data['negative'], data['predict_sentiment'] = zip(*data['tweet'].apply(lambda x:assign_tag(x)))
print(classification_report(data['sentiment'], data['predict_sentiment']))
'''
			precision    recall    f1-score    support

negative       0.32       0.46       0.37        13
neutral        0.66       0.48       0.56        60
positive       0.24       0.33       0.28        27

accuracy                             0.44        100
macro avg      0.41       0.43       0.40        100
weighted avg   0.50       0.44       0.46        100
'''
'''
improvement
translation
ml/dl model
'''
