# Sistema de Recomendação

# ***** Esta é a versão 2.0 deste script, atualizado em 02/07/2017 *****

import csv
import operator
import math
import numpy as np
from numpy import genfromtxt, savetxt
from sklearn import model_selection
from sys import argv
 
f = open("ratings.csv")
ratings_data = csv.reader(f)
listData = list(ratings_data)

f1 = open("toBeRated.csv")
toBeRated = csv.reader(f1)
toBeRatedList = list(toBeRated)

userRatings = {}

# Lista de usuários sem dicionário de filmes e sem ratings 
users = sorted(listData, key=operator.itemgetter(0))

# Mpapeando cada preferência de filme para cada usuário 
count = 0
u_prev = 0;
for u in users:
	userID = u[0]
	movieID = u[1]
	movieRating = u[2]
	if(u_prev == userID):
		userRatings[userID][movieID] = movieRating
		u_prev = userID
	else:
		userRatings[userID] = {}
		userRatings[userID][movieID] = movieRating
		u_prev = userID

# Mapeando cada filme avaliado por usuários e transformando o userRatings para item based  
def transposeRankings(ratings):
	transposed = {}
	for user in ratings:
		for item in ratings[user]:
			transposed.setdefault(item, {})
			transposed[item][user] = ratings[user][item]
	return transposed

# Calculando a similaridade usando Correlação Pearson 
def sim_pearson(ratings, user_1, user_2):
	similarity = {}
	for item in ratings[user_1]:
		if item in ratings[user_2]:
			similarity[item] = 1

	# Calculando o número de similaridades 
	numSim = len(similarity)

	# Se não houver similaridade, retorna 0 
	if numSim == 0:
		return 0

	# Adiciona o rating dado pelos usuários
	userOneSimArray = ([ratings[user_1][s] for s in similarity])
	userOneSimArray = map(int, userOneSimArray)

	sum_1 = sum(userOneSimArray)

	userTwoSimArray = ([ratings[user_2][s] for s in similarity])
	userTwoSimArray = map(int, userTwoSimArray)

	sum_2 = sum(userTwoSimArray)

	# Soma dos quadrados

	sum_1_sq = sum([pow(int(ratings[user_1][item]),2) for item in similarity])
	sum_2_sq = sum([pow(int(ratings[user_2][item]),2) for item in similarity])

	# Soma dos produtos (mutiplicação)
	productSum = sum([int(ratings[user_1][item]) * int(ratings[user_2][item]) for item in similarity])
	num = productSum - (sum_1*sum_2/numSim)
	den = math.sqrt((sum_1_sq - pow(sum_1,2)/numSim) * (sum_2_sq - pow(sum_2,2)/numSim))

	if den == 0:
		return 0
	r = num / den
	return r

# Calculando a similaridade Jaccard  
def compute_jaccard_similarity(ratings, user_1, user_2):
	# similarity = {}
	# for item in ratings[user_1]:
	# 	if item in ratings[user_2]:
	# 		similarity[item] = 1

	# numSim =len(similarity)

	# if numSim == 0:
	# 	return 0

	userOneRatingsArray = ([ratings[user_1][item] for item in ratings[user_1]])
	userOne = set(userOneRatingsArray)
	userTwoRatingsArray = ([ratings[user_2][item] for item in ratings[user_2]])
	userTwo = set(userTwoRatingsArray)

	return (len(set(userOne.intersection(userTwo))) / float(len(userOne.union(userTwo))))

# Calculando a similaridade Cosine  
def compute_cosine_similarity(ratings, user_1, user_2):
	similarity = {}
	for item in ratings[user_1]:
		if item in ratings[user_2]:
			similarity[item] = 1

	numSim =len(similarity)

	if numSim == 0:
		return 0

	userOneRatingsArray = ([ratings[user_1][s] for s in similarity])
	userOneRatingsArray = map(int, userOneRatingsArray)
	userTwoRatingsArray = ([ratings[user_2][s] for s in similarity])
	userTwoRatingsArray = map(int, userTwoRatingsArray)
	

	sum_xx, sum_yy, sum_xy = 0,0,0

	for i in range(len(userOneRatingsArray)):
		x = userOneRatingsArray[i]
		y = userTwoRatingsArray[i]

		sum_xx += x*x
		sum_yy += y*y
		sum_xy += x*y

	return sum_xy/math.sqrt(sum_xx*sum_yy)

# Calculando a similaridade 
def closeMatches(ratings, person, similarity):
	first_person = person
	scores = [(similarity(ratings, first_person, second_person), second_person) for second_person in ratings if second_person != first_person]
	scores.sort()
	scores.reverse()
	return scores


# Item based collaborative filtering 
def similarItems(ratings, similarity):
	itemList = {}

	itemsRatings = transposeRankings(ratings)
	c = 0
	for item in itemsRatings:
		c = c + 1
		# if c%100 == 0:
		# 	print "%d %d" % (c, len(itemsRatings))
		matches = closeMatches(itemsRatings, item, similarity)
		itemList[item] = matches
	return itemList


# Recomendações para uma pessoa, com base no peso dos ratings de outras pessoas 
def userBasedRecommendations(ratings, wantedPredictions, similarity):
	file = open('user.txt', 'a')
	ranks = {}

	for tuple in wantedPredictions:
		user = tuple[0]
		movieAsked = tuple[1]

		total = {}
		similaritySums = {}

		for second_person in ratings:
			if second_person == user: continue
			s = similarity(ratings, user, second_person)

			if s <= 0: continue

			for item in ratings[second_person]:
				if item not in ratings[user] or ratings[user][item] == 0:
					total.setdefault(item, 0)
					total[item] += int(ratings[second_person][item])*s
					similaritySums.setdefault(item, 0)
					similaritySums[item] += s
					ranks[item] = total[item]/similaritySums[item]
		file.write(str(ranks[movieAsked])+'\n')



def itemBasedRecommendations(ratings, itemToMatch, wantedPredictions):
	file = open('itemBasedRecos.txt', 'a')
	for tuple in wantedPredictions:
		user = tuple[0]
		movieAsked = tuple[1]

		uRatings = ratings[user]
		scores = {}
		total = {}
		ranks = {}


		# Itens avaliados por usuários
		for(item, rating) in uRatings.items():
		# Itens que são similares a esse
			for(similarity,item_2) in itemToMatch[item]:
			# Não considera se o usuário já avaliou este item
				if item_2 in uRatings: continue
				scores.setdefault(item_2, 0)
				scores[item_2] += similarity*int(rating)

				# Soma das similaridades
				total.setdefault(item_2,0)
				total[item_2] += similarity
				if total[item_2] == 0: 
					ranks[item_2] = 1
				else:
					ranks[item_2] = scores[item_2]/total[item_2]
		print (ranks[movieAsked])
		file.write(str(ranks[movieAsked]))

# Combinação de item based e user bases. Isso chama-se: Content - Boosted Collaborative Filtering
def itemBasedRecommendationsForCBCF(ratings, itemToMatch):
	for user in ratings:
		uRatings = ratings[user]
		scores = {}
		total = {}
		ranks = {}


		# Itens avaliados pelo usuário
		for(item, rating) in uRatings.items():
		# Itens que são similares a esse
			for(similarity,item_2) in itemToMatch[item]:
			# Não considera se o usuário já avaliou este item
				if item_2 in uRatings: 
					uRatings[item_2] = uRatings[item_2]
				else:
					scores.setdefault(item_2, 0)
					scores[item_2] += similarity*int(rating)

					# Soma das similaridades
					total.setdefault(item_2,0)
					total[item_2] += similarity
					if total[item_2] == 0: 
						uRatings[item_2] = 1
					else:
						uRatings[item_2] = scores[item_2]/total[item_2]
		
	return ratings

# Habilite os comandos abaixo, caso queira fazer algum teste específico
# simItems = similarItems(userRatings, compute_cosine_similarity)
# itemBasedRecommendations(userRatings, simItems, toBeRatedList)
# userBasedRecommendations(userRatings, toBeRatedList, compute_cosine_similarity)
# userBasedRecommendations(userRatings, '3371', sim_pearson)
# itemBasedReco = itemBasedRecommendationsForCBCF(userRatings, simItems)
# userRecosBasedOnDenseMatrix = userBasedRecommendations(itemBasedReco, toBeRatedList, compute_cosine_similarity)

def mainFunction():
	fileName = argv[1]
	similarityName = argv[2]
	print (similarityName)
	if(similarityName == 'cosine'): 
		sim = compute_cosine_similarity
	elif(similarityName == 'pearson'):
		sim = sim_pearson
	else:
		sim = compute_jaccard_similarity
	print (sim)
	userBasedRecommendations(userRatings, toBeRatedList, sim)


mainFunction()