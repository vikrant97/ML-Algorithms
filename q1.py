from csv import reader
import numpy as np
import sys
def load_csv_train(filename):
	dataset = list()
	labels=list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			dataset.append(row[1:])
			labels.append(row[0])
	dataset=np.asarray(dataset,int)
	labels=np.asarray(labels,int)
	return dataset,labels

def load_csv_test(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			dataset.append(row)
	dataset=np.asarray(dataset,int)
	return dataset


def score(actual, predicted):
	count=0
	tp=0;fp=0;fn=0
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

def predict(row, weights):
	activation = np.dot(row, weights)
	return 1 if activation>= 0 else 0

def simple_train_weights_without_margin(train_data, eta, max_itr, labels):
	weights = np.zeros(train_data.shape[1])
	count=0
	while True:
		#print count
		flag=0
		for i in range(train_data.shape[0]):
			prediction = predict(train_data[i],weights)
			error = labels[i] - prediction
			if error!=0:
				flag=1
				weights+=eta*error*train_data[i]
		if flag==0:
			break
		count+=1
		if count>max_itr:
			break
	return weights

def simple_train_weights_with_margin(train_data, eta, max_itr, b, labels):
	weights = np.zeros(train_data.shape[1])
	count=0
	while True:
		#print count
		flag=0
		for i in range(train_data.shape[0]):
			if labels[i]==1 and np.dot(weights,train_data[i])<=b:
				flag=1
				weights+=eta*train_data[i]
			if labels[i]==0 and np.dot(weights,train_data[i])>=-b:
				flag=1
				weights-=eta*train_data[i]
		if flag==0:
			break;
		count+=1
		if count>max_itr:
			break
	return weights

def batch_train_weights_without_margin(train_data, eta, max_itr, labels):
	weights = np.zeros(train_data.shape[1])
	count=0
	while True:
		#print count
		flag=0
		error_list = np.zeros(train_data.shape[1])
		for i in range(train_data.shape[0]):
			prediction = predict(weights,train_data[i])
			error = labels[i] - prediction
			if error!=0:
				flag=1
				error_list+=error*train_data[i]
		weights+=eta*error_list		
		if flag==0:
			break
		count+=1
		if count>max_itr:
			break
	return weights

def batch_train_weights_with_margin(train_data, eta, max_itr, b, labels):
	weights = np.zeros(train_data.shape[1])
	count=0
	while True:
		#print count
		flag=0
		error_list = np.zeros(train_data.shape[1])
		for i in range(train_data.shape[0]):
			if labels[i]==1 and np.dot(weights,train_data[i])<=b:
				flag=1
				error_list+=train_data[i]
			if labels[i]==0 and np.dot(weights,train_data[i])>=-b:
				flag=1
				error_list-=train_data[i]
		weights+=eta*error_list		
		if flag==0:
			break;
		count+=1
		if count>max_itr:
			break
	return weights

def test(weights, test_data,labels):
	predicted=list()
	actual=list()
	for i in range(test_data.shape[0]):
		prediction = predict(test_data[i], weights)
		print str(prediction)
		predicted.append(prediction)
		actual.append(labels[i])
	return actual,predicted

def main(train_file,test_file):
	train_dataset,labels=load_csv_train(train_file)
	test_dataset=load_csv_test(test_file)
	eta=1;max_itr=2500;b=10
	x=int(0.8*len(train_dataset))

	#Part1
	# weights1=simple_train_weights_without_margin(train_dataset[:x],eta,max_itr,labels[:x])	
	# actual1,predicted1=test(weights1,train_dataset[x:],labels[x:])
	# accuracy1,precision1,recall1=score(actual1,predicted1)
	# print "Simple perceptron without margin"
	# print "Accuracy: ", accuracy1
	# print "Precision: ",precision1
	# print "Recall: ",recall1
	weights1=simple_train_weights_without_margin(train_dataset,eta,max_itr,labels)	
	actual1,predicted1=test(weights1,test_dataset,labels) 

	#Part2
	# weights2=simple_train_weights_with_margin(train_dataset[:x],eta,max_itr,b,labels[:x])
	# actual2,predicted2=test(weights2,train_dataset[x:],labels[x:])
	# accuracy2,precision2,recall2=score(actual2,predicted2)
	# print "Simple perceptron with margin"
	# print "Accuracy: ",accuracy2
	# print "Precision: ",precision2
	# print "Recall: ",recall2
	weights2=simple_train_weights_with_margin(train_dataset,eta,max_itr,b,labels)
	actual2,predicted2=test(weights2,test_dataset,labels)

	#Part3
	# weights3=batch_train_weights_with_margin(train_dataset[:x],eta,max_itr,b,labels[:x])
	# actual3,predicted3=test(weights3,train_dataset[x:],labels[x:])
	# accuracy3,precision3,recall3=score(actual3,predicted3)
	# print "Batch perceptron with margin"
	# print "Accuracy: ",accuracy3
	# print "Precision: ",precision3
	# print "Recall: ",recall3
	weights2=batch_train_weights_with_margin(train_dataset,eta,max_itr,b,labels)
	actual2,predicted2=test(weights2,test_dataset,labels)
	
	#Part4
	# weights4=batch_train_weights_without_margin(train_dataset[:x],eta,max_itr, labels[:x])
	# actual4,predicted4=test(weights4,train_dataset[x:],labels[x:])
	# accuracy4,precision4,recall4=score(actual4,predicted4)
	# print "Batch perceptron without margin"
	# print "Accuracy: ",accuracy4
	# print "Precision: ",precision4
	# print "Recall: ",recall4
	weights2=batch_train_weights_without_margin(train_dataset,eta,max_itr, labels)
	actual2,predicted2=test(weights2,test_dataset,labels)

def scan():
	train_file=sys.argv[1]
	test_file=sys.argv[2]
	main(train_file,test_file)
scan()