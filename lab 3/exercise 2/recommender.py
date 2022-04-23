import os, pandas as pd

index_map = pd.read_csv('title', index_col='index')  # title map
matrix_files = os.listdir('cosine similarity')
print(index_map['title'].unique())
while True:
	while True:
		title = input('type the soundtrack title: ')
		if title == 'exit':
			exit()
		elif title in index_map['title'].unique():
			break
		else:
			print('we don’t have recommendations for %s, where %s is your option' % (title, title))
			print('please type again')
	index = index_map[index_map['title'] == title].index.tolist()[0]  # title index
	for file in matrix_files:
		start, end = int(file.split('-')[1]), int(file.split('-')[2])
		if start <= index < end:
			cosine_sim = pd.read_csv(os.path.join('cosine similarity', file), index_col='index')
			scores = cosine_sim.loc[index].sort_values(ascending=False)  # similarities for the input title
			indices = scores[1:11].index.to_list()
			for item in indices:
				print(index_map.loc[int(item), 'title'])