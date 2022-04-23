import json, re, string, pandas as pd, nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

with open('abbreviations.json', 'r', encoding='utf8')as fp:
	abbreviations = json.load(fp)
data = pd.read_json('digital music', lines=True).drop(columns=['reviewerID', 'asin', 'reviewerName', 'unixReviewTime'])[:1000]
data.loc[data[data['overall'] > 3].index, 'sentiment'] = 'positive'
data.loc[data[data['overall'] == 3].index, 'sentiment'] = 'neutral'
data.loc[data[data['overall'] < 3].index, 'sentiment'] = 'negative'

# text clean and tokenization
def convert_abbrev(word):
	return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
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
	tokens = word_tokenize(text)  # word segmentation
	tokens = [convert_abbrev(word) for word in tokens]  # stretch abbr
	text = ' '.join(tokens)
	return text

data['reviewText'] = data['reviewText'].apply(lambda x:string_handle(x.lower()))
data['summary'] = data['summary'].apply(lambda x:string_handle(x.lower()))
data['text length'] = (data['summary'] + ' ' + data['reviewText']).str.count(' ') + 1

data.to_csv('dataset', index=False)
