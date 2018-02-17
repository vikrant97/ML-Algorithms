from csv import reader
import numpy as np
import sys
def load_csv_train(filename):
	dataset = list()
	labels = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			flag=0
			for i in range(len(row)):
				if row[i]=='?':
					flag=1
			if not flag:
				dataset.append(row[1:len(row)-1])
				labels.append(row[len(row)-1])
	labels=np.asarray(labels,int)
	dataset=np.asarray(dataset,int)
	return dataset,labels

def load_csv_test(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			flag=0
			for i in range(len(row)):
				if row[i]=='?':
					flag=1
			if not flag:
				dataset.append(row[1:])
	dataset=np.asarray(dataset,int)
	return dataset

def score(actual, predicted):
	tp=0;fp=0;fn=0
	count=0
	for i in range(len(actual)):
		if actual[i]==predicted[i]:
			count+=1
		if actual[i]==4 and predicted[i]==4:
			tp+=1
		elif predicted[i]==4 and actual[i]==2:
			fp+=1
		elif predicted[i]==2 and actual[i]==4:
			fn+=1
	return count/float(len(actual)),tp/float(tp+fp),tp/float(tp+fn)

def predict(row, weights):
	activation = np.dot(row, weights)
	return 4 if activation>= 0 else 2

def simple_relax_train_weights_with_margin(train_data, labels, eta, max_itr,b):
	weights = np.zeros(train_data.shape[1])
	count=0
	while True:
		# print count
		flag=0
		for i in range(train_data.shape[0]):
			wx=np.dot(weights,train_data[i])
			norm=np.dot(train_data[i], train_data[i])
			if labels[i]==4 and wx<=b:
				flag=1
				weights+=eta*((b-wx)/norm)*train_data[i]
			if labels[i]==2 and wx>=-b:
				flag=1
				weights-=eta*((wx+b)/norm)*train_data[i]
		if flag==0:
			break
		count+=1
		if count>max_itr:
			break;
	return weights

def modified_perceptron_train_weights_with_margin(train_data, labels, eta, max_itr,b):
	weights = np.zeros(train_data.shape[1])
	itr=0
	while True:
		#print count
		count=0
		flag=0
		error_list = np.zeros(train_data.shape[1])
		for i in range(train_data.shape[0]):
			wx=np.dot(weights,train_data[i])
			norm=np.dot(train_data[i], train_data[i])
			if labels[i]==4 and wx<=b:
				flag=1
				error_list+=train_data[i]
				count+=1
			if labels[i]==2 and wx>=-b:
				flag=1
				error_list-=train_data[i]
				count+=1
		# weights+=(1/float(count**2))*eta*error_list
		if (count/float(train_data.shape[0]))*100>=5:
			weights+=(1/float(count**2))*eta*error_list
		else:
			break		
		if flag==0:
			break;
		itr+=1
		if itr>max_itr:
			break
	return weights

# def modified_perceptron_train_weights_with_margin(train_data, labels, eta, max_itr,b):
# 	weights = np.zeros(train_data.shape[1])
# 	total=train_data.shape[1]
# 	weighted=np.zeros(train_data.shape[1])
# 	itr=0
# 	while True:
# 		flag=0
# 		error_list = np.zeros(train_data.shape[1])
# 		for i in range(train_data.shape[0]):
# 			count=0
# 			wx=np.dot(weights,train_data[i])
# 			norm=np.dot(train_data[i], train_data[i])
# 			if labels[i]==4 and wx<=b:
# 				flag=1
# 				error_list+=train_data[i]
# 				count+=1
# 			if labels[i]==2 and wx>=-b:
# 				flag=1
# 				error_list-=train_data[i]
# 				count+=1
# 		weighted+=((total-count)/total)*eta*weights
# 		weights+=eta*error_list
# 		if flag==0:
# 			break;
# 		itr+=1
# 		if itr>max_itr:
# 			break
# 	return weighted

def test(weights, test_data,labels):
	predicted=list()
	actual=list()
	for i in range(test_data.shape[0]):
		prediction = predict(test_data[i], weights)
		print prediction
		predicted.append(prediction)
		actual.append(labels[i])
	return (actual,predicted)

def main(train_file,test_file):
	eta=2;max_itr=5000;b=10
	train_dataset,labels=load_csv_train(train_file)
	test_dataset=load_csv_test(test_file)
	x=int(0.8*len(train_dataset))

	###PART1
	# weights1=simple_relax_train_weights_with_margin(train_dataset[0:x],labels[0:x],eta,max_itr,b)
	# actual1,predicted1=test(weights1,train_dataset[x:], labels[x:])
	# accuracy1, precision1, recall1= score(actual1,predicted1)
	# print "Simple relaxed perceptron with margin"
	# print "Accuracy: ", accuracy1
	# print "Precision: ",precision1
	# print "Recall: ",recall1
	weights1=simple_relax_train_weights_with_margin(train_dataset,labels,eta,max_itr,b)
	actual1,predicted1=test(weights1,test_dataset, labels)
	
	###PART2
	# weights2=modified_perceptron_train_weights_with_margin(train_dataset[:x],labels[:x],eta,max_itr,b)
	# actual2,predicted2=test(weights2,train_dataset[x:], labels[x:])
	# accuracy2, precision2, recall2= score(actual2,predicted2)
	# print "Modified perceptron with margin"
	# print "Accuracy: ",accuracy2
	# print "Precision: ",precision2
	# print "Recall: ",recall2
	weights2=modified_perceptron_train_weights_with_margin(train_dataset,labels,eta,max_itr,b)
	actual2,predicted2=test(weights2,test_dataset, labels)
	
def scan():
	train_file=sys.argv[1]
	test_file=sys.argv[2]
	main(train_file,test_file)
scan()