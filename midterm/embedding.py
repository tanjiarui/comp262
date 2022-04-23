from gensim.models import Word2Vec

corpus = [['girl', 'kicks', 'cat'], ['cat', 'kicks', 'girl'], ['girl', 'eats', 'food'], ['cat', 'eats', 'fish']]
# skip gram
skip_gram = Word2Vec(corpus, min_count=1, sg=1)
# summarize vocabulary
words = list(skip_gram.wv.index_to_key)
print(words)
# most similar word
for word in words:
	print(skip_gram.wv.most_similar(word))
# pair similarity
print('similarity between cat and girl:', skip_gram.wv.similarity('cat', 'girl'))
print('similarity between eats and food:', skip_gram.wv.similarity('eats', 'food'))
print('similarity between girl and eats:', skip_gram.wv.similarity('girl', 'eats'))

# CBOW
cbow = Word2Vec(corpus, min_count=2, sg=0)
# summarize vocabulary
words = list(cbow.wv.index_to_key)
print(words)
# most similar word
for word in words:
	print(cbow.wv.most_similar(word))
# pair similarity
print('similarity between cat and girl:', cbow.wv.similarity('cat', 'girl'))
print('similarity between girl and eats:', cbow.wv.similarity('girl', 'eats'))