#!/usr/bin/env python
###################################
# CS B551 Spring 2017, Assignment #4
#
# Your names and user ids: 
# Kwangwuk Lee, lee2074
# Xueqiang Wang, xw48
#
# Part 1: In this part, we implemented a nearest neighbor classifier. For each test item, we calculate its nearest neighbor 
# (based on Manhattan Distance of two vectors) and predict its label. 
#
# We display the confusion matrix, including precision and recall of each class (i.e., 0, 90, 180, 270).
# The result is as follows: (./orient.py nearest train-data.txt test-data.txt)
# 
# Accuracy 0.679745493107
#
# =========================================================================
# Matrix:
#              		Real_0		Real_90		Real_180		Real_270
# Predicted_0  		153			14			45				19
# Predicted_90 		23			159			21				33
# Predicted_180		41			16			154				17
# Predicted_270		22			35			16				175

# =========================================================================
#          		Recall		Precision
# Class_0  		0.640167		0.662338
# Class_90 		0.709821		0.673729
# Class_180		0.652542		0.675439
# Class_270		0.717213		0.705645
#
# Part 2: In this part, we built a neural network classifier. The neural network is fully-connected with 192 input nodes, 
# 4 output nodes and several hidden layers (configured by NUM_HIDDEN_LAYER and NUM_HIDDEN_NODES below). As for activation 
# function, we tried Sigmoid function and TanH, and chose Sigmoid in code (as evaluated, it converage faster for our dataset). 
# We also implement the backward propagation to train the neural network model. Learning rate can be configured by LEARNING_RATE. 
#
# Before testing or training, all features are normalized to range (0, 1.0) and label are extended to a vector (0->[1,0,0,0], 
# 90->[0,1,0,0], 180->[0,0,1,0], 270->[0,0,0,1]).
#
# As for the experiment, we use the smaller dataset (test-data.txt) as training set, meanwhile we use larger one (train-data.txt) 
# as testing. The reason is that building model is slower, while predicting label on neural network is much more faster. In this part,
# we measure how learning rate and number of nodes per hidden layer affects performance. And the mean of error and classification 
# result are given below:
#  Measuring mean error when training model
#			Learning Rate  		0.001		0.002		0.003		0.004		0.005		0.006		0.007		0.008		0.009	
# nodes
#  5							0.073		0.058		0.050		0.074		0.064		0.065		0.054		0.053		0.052
#  15							0.038		0.021		0.018		0.018		0.019		0.018		0.016		0.013		0.011
#  25							0.025		0.014		0.012		0.012		0.011		0.010		0.008		0.010		0.009
#  35							0.024		0.015		0.012		0.009		0.009		0.008		0.007		0.008		0.006
#  45							0.021		0.013		0.010		0.008		0.009		0.007		0.009		0.007		0.008	
#  55							0.020		0.011		0.008		0.009		0.008		0.008		0.008		0.008		0.007	
#  65							0.019		0.011		0.010		0.008		0.007		0.006		0.005		0.007		0.007	
#  75							0.017		0.011		0.008		0.008		0.007		0.007		0.007		0.006		0.007
#
#  Discuss learning rate: 1) as shown in above table, error increases when learning rate leads to a local optimal solution (As of nodes=5, learning rate=0.004). 
#						  2) when learning rate is smaller, neural network converages slower.
#
#	Measuring accuracy when predicting label
#			Learning Rate  		0.001		0.002		0.003		0.004		0.005		0.006		0.007		0.008		0.009	
# nodes
# 5								0.6914		0.6817		0.6776		0.6660		0.6759		0.6761		0.6884		0.6902		0.6892
# 15							0.6869		0.6801		0.6793		0.6796		0.6785		0.6716		0.6771		0.6798		0.6765
# 25							0.6914		0.6884		0.6884		0.6878		0.6811		0.6844		0.6824		0.6847		0.6777
# 35							0.6924		0.6908		0.6930		0.6882		0.6902		0.6905		0.6879		0.6982		0.6898
# 45							0.6917		0.6915		0.6901		0.6937		0.6897		0.6933		0.6947		0.6924		0.6961
# 55							0.6912		0.6893		0.6908		0.6907		0.6964		0.6970		0.6970		0.6938		0.7020							
# 65							0.6932		0.6934		0.6965		0.6945		0.6986		0.6955		0.7000		0.6950		0.6960
# 75							0.6961		0.6964		0.7010		0.7018		0.7019		0.6989		0.7015		0.7011		0.7040
#
#	Recommendation of parameters: 1) as shown above, i would recommend 45 to number of nodes of hidden layer. And 0.004 as learning rate. This learning
#									rate is smaller, and may avoid some local optima.
# 
#	Mearsuing running time: To give an understanding of how fast neural network parameter propagates, we set iterations of pass to a same value (10000), and
#   modify nodes number of per hidden layer. The time consumption is given below:
#			Nodes per hidden layer 		5			15			25			35			45			55
#			Time in seconds				34.73		53.16		75.20		94.83		120.73		141.11
#
#			In our program, back propagation and gradient descent are mostly based on matrix operation. When number of nodes per hidden layer increases, the time
#			spent on matrix operations increases. (as shown above). Similarly, when training set increases, the size of input matrix and correspondingly the running 
#			time would increase. 
#    
#	Observed incorrect prediction: https://www.flickr.com/photos/enfocalafoca/10008667845
#									This image is always classified as "180". It may be caused by overfitting. However, this image is rather tricky. It looks like a
#									picture from any orientation.
#
# The confusion matrix is as follows:
# Accuracy 0.693747295543
#
# =========================================================================
# Matrix:
#              		Real_0		Real_90		Real_180		Real_270
# Predicted_0  		6190		790		1251		752
# Predicted_90 		654		6422		754		1092
# Predicted_180		1637		932		6593		953
# Predicted_270		763		1100		646		6447
#
# =========================================================================
#         		Recall		Precision
# Class_0  		0.669624		0.689079
# Class_90 		0.694721		0.719794
# Class_180		0.713219		0.651804
# Class_270		0.697425		0.719853
####


import sys
import math
from heapq import heappush, heappop
import numpy as np
import os.path
import time

def read_dataset(f_path):
	if not os.path.isfile(f_path):
		print f_path, "does not exist, exiting..."
		sys.exit()
	f_file = open(f_path, 'r')
	dataset = []
	for line in f_file:
		x = line.split()
		vec = [int(w) for w in x[1:]]
		vec.insert(0, x[0])
		dataset.append(vec)
	return dataset

def manhattan_distance(start, end):
	dist = 0
	for i in range(0, len(start)):
		dist += abs(start[i] - end[i])
	return dist

def show_matrix(real_label, predicted_label):
	predicted_0_real_0 = sum(predicted_label[i] == 0 and real_label[i] == 0 for i in range(0, len(real_label)))
	predicted_0_real_90 = sum(predicted_label[i] == 0 and real_label[i] == 90 for i in range(0, len(real_label)))
	predicted_0_real_180 = sum(predicted_label[i] == 0 and real_label[i] == 180 for i in range(0, len(real_label)))
	predicted_0_real_270 = sum(predicted_label[i] == 0 and real_label[i] == 270 for i in range(0, len(real_label)))

	predicted_90_real_0 = sum(predicted_label[i] == 90 and real_label[i] == 0 for i in range(0, len(real_label)))
	predicted_90_real_90 = sum(predicted_label[i] == 90 and real_label[i] == 90 for i in range(0, len(real_label)))
	predicted_90_real_180 = sum(predicted_label[i] == 90 and real_label[i] == 180 for i in range(0, len(real_label)))
	predicted_90_real_270 = sum(predicted_label[i] == 90 and real_label[i] == 270 for i in range(0, len(real_label)))

	predicted_180_real_0 = sum(predicted_label[i] == 180 and real_label[i] == 0 for i in range(0, len(real_label)))
	predicted_180_real_90 = sum(predicted_label[i] == 180 and real_label[i] == 90 for i in range(0, len(real_label)))
	predicted_180_real_180 = sum(predicted_label[i] == 180 and real_label[i] == 180 for i in range(0, len(real_label)))
	predicted_180_real_270 = sum(predicted_label[i] == 180 and real_label[i] == 270 for i in range(0, len(real_label)))

	predicted_270_real_0 = sum(predicted_label[i] == 270 and real_label[i] == 0 for i in range(0, len(real_label)))
	predicted_270_real_90 = sum(predicted_label[i] == 270 and real_label[i] == 90 for i in range(0, len(real_label)))
	predicted_270_real_180 = sum(predicted_label[i] == 270 and real_label[i] == 180 for i in range(0, len(real_label)))
	predicted_270_real_270 = sum(predicted_label[i] == 270 and real_label[i] == 270 for i in range(0, len(real_label)))

	print "Accuracy", (float)(predicted_0_real_0 + predicted_90_real_90 + predicted_180_real_180 + predicted_270_real_270)/(float)(len(real_label))
	print "\n========================================================================="
	print "Matrix:"
	print "             \t\tReal_0\t\tReal_90\t\tReal_180\t\tReal_270"
	print "Predicted_0  \t\t%d\t\t%d\t\t%d\t\t%d" % (predicted_0_real_0, predicted_0_real_90, predicted_0_real_180, predicted_0_real_270)
	print "Predicted_90 \t\t%d\t\t%d\t\t%d\t\t%d" % (predicted_90_real_0, predicted_90_real_90, predicted_90_real_180, predicted_90_real_270)
	print "Predicted_180\t\t%d\t\t%d\t\t%d\t\t%d" % (predicted_180_real_0, predicted_180_real_90, predicted_180_real_180, predicted_180_real_270)
	print "Predicted_270\t\t%d\t\t%d\t\t%d\t\t%d" % (predicted_270_real_0, predicted_270_real_90, predicted_270_real_180, predicted_270_real_270)

	print "\n========================================================================="
	print "         \t\tRecall\t\tPrecision"
	print "Class_0  \t\t%f\t\t%f" % ((float)(predicted_0_real_0)/(float)(predicted_0_real_0 + predicted_90_real_0 + predicted_180_real_0 + predicted_270_real_0), (float)(predicted_0_real_0)/(float)(predicted_0_real_0 + predicted_0_real_90 + predicted_0_real_180 + predicted_0_real_270))
	print "Class_90 \t\t%f\t\t%f" % ((float)(predicted_90_real_90)/(float)(predicted_0_real_90 + predicted_90_real_90 + predicted_180_real_90 + predicted_270_real_90), (float)(predicted_90_real_90)/(float)(predicted_90_real_0 + predicted_90_real_90 + predicted_90_real_180 + predicted_90_real_270))
	print "Class_180\t\t%f\t\t%f" % ((float)(predicted_180_real_180)/(float)(predicted_0_real_180 + predicted_90_real_180 + predicted_180_real_180 + predicted_270_real_180), (float)(predicted_180_real_180)/(float)(predicted_180_real_0 + predicted_180_real_90 + predicted_180_real_180 + predicted_180_real_270))
	print "Class_270\t\t%f\t\t%f" % ((float)(predicted_270_real_270)/(float)(predicted_0_real_270 + predicted_90_real_270 + predicted_180_real_270 + predicted_270_real_270), (float)(predicted_270_real_270)/(float)(predicted_270_real_0 + predicted_270_real_90 + predicted_270_real_180 + predicted_270_real_270))

def nearest(argv):
	test_set = read_dataset(sys.argv[3])
	train_set = read_dataset(sys.argv[2])

	real_label = [row[1] for row in test_set]
	predicted_label = []

	finished = 0
	for test_item in test_set:
		print "processing test data #", finished
		finished += 1
		nearest_item = (-1, -1)

		for train_idx in range(0, len(train_set)):
			dist = manhattan_distance(test_item[2:], train_set[train_idx][2:])
			if nearest_item[1] > dist or nearest_item[0] < 0:
				nearest_item = (train_idx, dist)

		predicted_label.append(train_set[nearest_item[0]][1])

	show_matrix(real_label, predicted_label)

	ofile = open('nearest_output.txt', 'w')
	for idx in range(0, len(test_set)):
		ofile.write("%s %s\n" % (test_set[idx][0], predicted_label[idx]))
	ofile.close()

# activation function: sigmoid
def activation(x):
	#return np.tanh(x)
	return 1/(1+np.exp(-x))

# derivative of activation function
def derivative(x):
	#return 1-np.power(np.tanh(x), 2)
	return x*(1-x)

def data_normalize(data_set):
	np_feature = []
	np_label = []
	for v in data_set: 
		if v[1] == 0:
			np_label.append([1,0,0,0])
		elif v[1] == 90:
			np_label.append([0,1,0,0])
		elif v[1] == 180:
			np_label.append([0,0,1,0])
		elif v[1] == 270:
			np_label.append([0,0,0,1])

		np_feature.append([float(w)/256.0 for w in v[2:]])
	return (np.array(np_feature), np.array(np_label))

def nnet_train(argv, NUM_HIDDEN_LAYER, NUM_HIDDEN_NODES, LEARNING_RATE):
	model_output = argv[3]

	print 'Parameter', NUM_HIDDEN_LAYER, NUM_HIDDEN_NODES, LEARNING_RATE

	print "reading training data..."
	raw_train_set = read_dataset(argv[2])
	#normalize training data
	np_feature, np_label = data_normalize(raw_train_set)

	np.random.seed(1)
	# max number of iterations
	MAX_ITERATIONS = 60000

	# initialize weights for each layer
	weights = []
	# weight of input to hidden layer
	weights.append(2*np.random.random((192, NUM_HIDDEN_NODES)) - 1)
	for idx in range(0, NUM_HIDDEN_LAYER-1):
		weights.append(2*np.random.random((NUM_HIDDEN_NODES, NUM_HIDDEN_NODES)) - 1)
	# weight of hidden layer to output
	weights.append(2*np.random.random((NUM_HIDDEN_NODES,4)) - 1)

	start = time.time()

	for it in xrange(MAX_ITERATIONS):
		previous_layer = np_feature

		layer_state = []
		layer_state.append(previous_layer)

		for layer_idx in range(0, len(weights)):
			previous_layer = activation(np.dot(previous_layer, weights[layer_idx]))
			layer_state.append(previous_layer)
		#print layer_state

		output_error = np_label - layer_state[len(layer_state)-1]
		output_error_mean = np.mean(np.abs(output_error))

		if it%5000 == 0:
			print "current error (mean):", output_error_mean
			print "training..."
		# back propagation
		next_layer_error = output_error
		for idx in range(len(weights)-1, -1, -1):
			delta = next_layer_error*derivative(layer_state[idx+1])
			next_layer_error = delta.dot(weights[idx].T)
			weights[idx] += LEARNING_RATE*layer_state[idx].T.dot(delta)

	stop = time.time()
	print "running time in seconds: ", stop-start

	print "writing model to file..."
	#o_f = open(model_output+'_'+str(NUM_HIDDEN_LAYER)+'_'+str(NUM_HIDDEN_NODES)+'_'+str(LEARNING_RATE), "w")
	o_f = open(model_output, "w")
	np.save(o_f, weights)
	o_f.close()

def nnet_test(argv):
	model_input = argv[3]

	print "loading model from file..."
	i_f = open(model_input, "r")
	weights = np.load(i_f)
	i_f.close()

	print "reading testing data..."
	raw_test_set = read_dataset(argv[2])
	#normalize testing data
	np_feature, np_label = data_normalize(raw_test_set)

	real_label = [w[1] for w in raw_test_set]
	predicted_label = []

	# calculate on trained model
	input_data = np_feature
	for idx in range(0, len(weights)):
		output_data = activation(np.dot(input_data, weights[idx]))
		input_data = output_data

	for k in output_data:
		l = k.tolist()
		predicted_label.append(90*l.index(max(l)))

	show_matrix(real_label, predicted_label)

	ofile = open('nnet_output.txt', 'w')
	for idx in range(0, len(raw_test_set)):
		ofile.write("%s %s\n" % (raw_test_set[idx][0], predicted_label[idx]))
	ofile.close()


def usage():
	print "Usage: choose one of "
	print "    ./orient.py nearest train_file.txt test_file.txt"
	print "    ./orient.py nnet train_file.txt model_file.txt"
	print "    ./orient.py nnet test_file.txt model_file.txt"
	sys.exit()

if __name__ == "__main__":
	if len(sys.argv) != 4:
		usage()

	part = sys.argv[1]
	if part == "nearest":
		nearest(sys.argv)
	elif part == "nnet":
		model_file_name = sys.argv[3]

		if os.path.isfile(model_file_name):
			# if model file exists
			print "model file does exist, start testing..."
			nnet_test(sys.argv)
		else:
			print "model file does NOT exist, start training..."
			# number of hidden layers
			NUM_HIDDEN_LAYER = 1
			# number of nodes per hidden layer
			NUM_HIDDEN_NODES = 45
			# learning rate
			LEARNING_RATE = 0.004
			nnet_train(sys.argv, NUM_HIDDEN_LAYER, NUM_HIDDEN_NODES, LEARNING_RATE)

			#for NUM_HIDDEN_LAYER in range(1, 3):
			#	for NUM_HIDDEN_NODES in range(5, 200, 10):
			#		for LEARNING_RATE in np.arange(0.001,0.01,0.001):
			#			nnet_train(sys.argv, NUM_HIDDEN_LAYER, NUM_HIDDEN_NODES, LEARNING_RATE)
						#for NUM_HIDDEN_LAYER in range(1, 3):
	else:
		print "algorithm is not correctly specified, exiting..."
		sys.exit()