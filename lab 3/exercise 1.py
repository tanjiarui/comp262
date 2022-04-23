import pandas as pd
from apyori import apriori

# load and eda
data = pd.read_json('recipes.json')
print('number of instances: %d' % data.shape[0])
print('type of cuisines: %d' % data['cuisine'].unique().size)
print('number of recipes within each type:')
print(data.groupby('cuisine', as_index=False)['id'].count().rename(columns={'id': 'recipes'}))

while True:
	# choose cuisine
	while True:
		cuisine = input('type your cuisine: ')
		if cuisine in data['cuisine'].unique():
			break
		elif cuisine == 'exit':
			exit()
		else:
			print('we don’t have recommendations for %s, where %s is your option' % (cuisine, cuisine))
			print('please type again')
	# analyze ingredients
	ingredients = data[data['cuisine'] == cuisine]['ingredients']
	association_rules = apriori(ingredients, min_support=100/ingredients.size, min_confidence=.5)
	result = dict()
	for item in association_rules:
		if len(item[0]) < 2:
			continue
		for stat in item[2]:
			base_list = list(stat[0])
			# if base item set is empty then go to the next record
			if not base_list:
				continue
			base_list.sort()
			base_key = tuple(base_list)
			if base_key not in result.keys():
				result[base_key] = list()
			result[base_key].append([list(stat[1])[0], stat[3]])
	# sort the rules in descending order of lift values
	for rule_list in result:
		result[rule_list].sort(key=lambda x: x[1], reverse=True)
	for base_item, rules in result.items():
		# lift greater than 2
		for rule in rules:
			if rule[1] > 2:
				print('{} -> {}\tlift: {:f}'.format(rule[0], ' '.join(base_item), rule[1]))
'''
number of instances: 39774
type of cuisines: 20
number of recipes within each type:
         cuisine  recipes
0      brazilian      467
1        british      804
2   cajun_creole     1546
3        chinese     2673
4       filipino      755
5         french     2646
6          greek     1175
7         indian     3003
8          irish      667
9        italian     7838
10      jamaican      526
11      japanese     1423
12        korean      830
13       mexican     6438
14      moroccan      821
15       russian      489
16   southern_us     4320
17       spanish      989
18          thai     1539
19    vietnamese      825
'''