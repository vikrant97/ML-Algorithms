#!/usr/bin/env python
import sys
import os
import numpy as np
import math,operator
 
classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']

def extract_features(traindir):
	vocab=set()
	#stop_words=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "eicept", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "neit", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "sii", "siity", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
	#stop_words = ["what","who","is","a","at","is","he","the"]
	for c in classes:
		listing = os.listdir(traindir+c)
		for filename in listing:
			f = open(traindir+c+filename,'r')
			inputs = f.read()
			f.close()
			words_list=inputs.split()
			#non_stop_words_list=[word for word in words_list if word not in stop_words]
			unique_words=set(words_list)
			vocab.update(unique_words)
	return vocab

def makeFeatureVector(inputdir):
	trainSet=list()
	testSet=list()
	for idir in inputdir:
		classid = 0
		for c in classes:
			listing = os.listdir(idir+c)
			for filename in listing:
				d={}
				for w in vocab:
					d[w]=0
				f = open(idir+c+filename,'r')
				inputs = f.read()
				f.close()
				for w in inputs:
					if w in vocab:
						d[w]+=1
				feature_vector=[]
				for w in vocab:
					feature_vector.append(d[w])
				y=((classes[classid]).strip('/'))
				feature_vector.append(y)
				if idir == traindir:
					trainSet.append(feature_vector)
				else:
				 	testSet.append(feature_vector)
			classid += 1
	return trainSet,testSet

def accuracy(actual,predicted):
	count=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			count+=1
	return count/float(len(actual))

def euclideanDistance(vec1, vec2, length):
	distance = 0
	for i in range(length):
		distance += (abs(vec1[i] - vec2[i]))
	return (distance)

def KNN(trainingSet, testVec, k):
	neighbors = []
	distances = []
	for i in range(len(trainingSet)):
		distance = euclideanDistance(testVec, trainingSet[i], len(testVec)-1)
		distances.append((trainingSet[i], distance))
	distances.sort(key=operator.itemgetter(1))
	for i in range(k):
		neighbors.append(distances[i][0])
	return neighbors

def best_class(neighbors):
	count = {}
	for i in range(len(neighbors)):
		class_label = neighbors[i][-1]
		if class_label not in count:
			count[class_label] = 1
		else:
			count[class_label] += 1
	count = sorted(count.iteritems(), key=operator.itemgetter(1), reverse=True)
	return count[0][0]

def KNN_classify(trainSet,testSet,k):
	predicted=[]
	actual=[]
	for i in range(len(testSet)):
		testVec=testSet[i]
		neighbors=KNN(trainSet,testVec,k)
		prediction=best_class(neighbors)
		predicted.append(prediction)
		actual.append(testVec[-1])
		print prediction
	#print accuracy(actual,predicted)	

if __name__ == '__main__':
	traindir = sys.argv[1]
	testdir = sys.argv[2]
	inputdir = [traindir,testdir]
	#Parameters
	k=10

	#Extract features
	vocab=extract_features(traindir)	
	#print len(vocab)
	
	#print('Making the feature vectors.')
	trainSet,testSet=makeFeatureVector(inputdir)

	#print('Classification started.')
	KNN_classify(trainSet,testSet,k)	
