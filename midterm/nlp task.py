import spacy

text = 'Wherever you are from, and wherever you would like to be, SaGE would like to help you broaden your horizons, build your global network, and achieve academic, personal, and professional success during your stay at Centennial.'


def get_nouns(text):
	nlp = spacy.load('en_core_web_sm')
	doc = nlp(text)  # converting the text into a spacy doc
	for token in doc:
		if token.pos_ in ['NOUN', 'PROPN']:
			print(token, ':', token.pos_)


get_nouns(text)