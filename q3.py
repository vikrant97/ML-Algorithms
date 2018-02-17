from csv import reader
import numpy as np
import sys
import math
def load_csv_train(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		i=0
		for row in csv_reader:
			if i!=0:
				if row[9]=='low':
					row[9]=0
				elif row[9]=='medium':
					row[9]=1
				elif row[9]=='high':
					row[9]=2
				if row[8]=='accounting':
					row[8]=0
				elif row[8]=='hr':
					row[8]=1
				elif row[8]=='IT':
					row[8]=2
				elif row[8]=='management':
					row[8]=3
				elif row[8]=='marketing':
					row[8]=4
				elif row[8]=='product_mng':
					row[8]=5
				elif row[8]=='RandD':
					row[8]=6
				elif row[8]=='sales':
					row[8]=7
				elif row[8]=='support':
					row[8]=8
				elif row[8]=='technical':
					row[8]=9
				dataset.append(row)
			else:
				i=1
			
	dataset=np.asarray(dataset,float)
	return dataset

def load_csv_test(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		i=0
		for row in csv_reader:
			if i!=0:
				if row[8]=='low':
					row[8]=0
				elif row[8]=='medium':
					row[8]=1
				elif row[8]=='high':
					row[8]=2
				if row[7]=='accounting':
					row[7]=0
				elif row[7]=='hr':
					row[7]=1
				elif row[7]=='IT':
					row[7]=2
				elif row[7]=='management':
					row[7]=3
				elif row[7]=='marketing':
					row[7]=4
				elif row[7]=='product_mng':
					row[7]=5
				elif row[7]=='RandD':
					row[7]=6
				elif row[7]=='sales':
					row[7]=7
				elif row[7]=='support':
					row[7]=8
				elif row[7]=='technical':
					row[7]=9
				dataset.append(row)
			else:
				i=1
			
	dataset=np.asarray(dataset,float)
	return dataset

def metrics(actual, predicted):
	tp=0;fp=0;fn=0
	count=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			count+=1
		if actual[i]==1 and predicted[i]==1:
			tp+=1
		elif predicted[i]==1 and actual[i]==0:
			fp+=1
		elif predicted[i]==0 and actual[i]==1:
			fn+=1
	return count/float(len(actual)),tp/float(tp+fp),tp/float(tp+fn)

# def calc_score(groups, classes):
# 	score = 0.0
# 	total = float(sum([len(group) for group in groups]))
# 	for group in groups:
# 		p1=0;p2=0
# 		if len(group) == 0:
# 			continue
# 		for row in group:
# 			if row[6]==0:
# 				p1+=1
# 			else:
# 				p2+=1
# 		p1=p1/len(group); p2=p2/len(group)
# 		p1=p1**2; p2=p2**2
# 		score += (1.0 - p1-p2) * (len(group) / total)
# 	return score


def calc_score(groups, classes):
	score = 0.0
	total = float(sum([len(group) for group in groups]))
	for group in groups:
		p=0;n=0
		if len(group) != 0:
			for row in group:
				if row[6]==0.0:
					n+=1
				else:
					p+=1
			p=p/len(group); n=n/len(group)
			if p!=0 and n!=0:
				p=-1*p*(math.log(p)/math.log(2));n=-1*n*(math.log(n)/math.log(2))
				score += (p+n) * (len(group) / total)
			else:
				score-=1
	return score

def binary_construct(index, value, dataset):
	left=[];right=[]
	for row in dataset:
		if row[index] >= value:
			right.append(row)
		else:
			left.append(row)
	return left,right

def generate_node(dataset):
	max_score=100000.0
	classes = [0,1]
	max_index=0
	max_value=0.5
	max_groups=[[],[]]
	for index in range(len(dataset[0])):
		if index!=6:
			unique_dataset=list(set(row[index] for row in dataset))
			for i in range(len(unique_dataset)):
				groups=binary_construct(index, unique_dataset[i], dataset)
				score = calc_score(groups, classes)
				if score < max_score:
					max_score=score
					max_index=index
					max_value=unique_dataset[i]
					max_groups=groups
					
	return {'index':max_index, 'value':max_value, 'groups':max_groups}

def create_terminal(group):
	p=0
	n=0
	for row in group:
		if row[6]==1:
			p+=1
		else:
			n+=1
	if p>n:
		return 1.0
	return 0.0

def construct(node, max_depth, min_size, depth):
	left, right = node['groups']
	#print depth
	del(node['groups'])

	if depth > max_depth:
		node['left']=create_terminal(left) 
		node['right']=create_terminal(right)
		return

	if not left:
		node['left'] = create_terminal(right)
		node['right'] = create_terminal(right)
		return
	if not right:
		node['right'] = create_terminal(left)
		node['left'] = create_terminal(left)
		return

	if len(right) <= min_size:
		node['right'] = create_terminal(right)
	else:
		node['right'] = generate_node(right)	
		construct(node['right'], max_depth, min_size, depth+1)
	
	if len(left) <= min_size:
		node['left'] = create_terminal(left)
	else:
		node['left'] = generate_node(left)
		construct(node['left'], max_depth, min_size, depth+1)

def build_tree(train_dataset, max_depth, min_size):
	root = generate_node(train_dataset)
	construct(root, max_depth, min_size, 0)
	return root

def predict(node, row):
	if row[node['index']] >= node['value']:
		if isinstance(node['right'], float):
			return node['right']
		else:
			return predict(node['right'], row)
	else:
		if isinstance(node['left'], float):
			return node['left']
		else:
			return predict(node['left'], row)

def test(test_dataset,tree):
	predicted = list()
	actual= list()
	for row in test_dataset:
		prediction = predict(tree, row)
		predicted.append(prediction)
		actual.append(row[6])
		print str(int(prediction))	
	return actual,predicted
			
def main():
	max_depth=15;min_size=10
	train_file=sys.argv[1]; test_file=sys.argv[2]

	train_dataset=load_csv_train(train_file); 
	test_dataset=load_csv_test(test_file)

	tree=build_tree(train_dataset,max_depth,min_size)
	actual,predicted=test(test_dataset,tree)

	# x=int(0.8*len(train_dataset))
	# tree=build_tree(train_dataset[:x],max_depth,min_size)
	# actual,predicted=test(train_dataset[x:],tree)
	# accuracy,precision,recall=metrics(actual,predicted)
	# print 'Accuracy: ',accuracy
	# print 'Precision: ',precision
	# print 'Recall: ',recall 

main()
