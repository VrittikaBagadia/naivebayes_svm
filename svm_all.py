from datetime import datetime
import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import scipy
from scipy.spatial.distance import pdist, squareform
import math
from svmutil import *
import sys

def read_data():
	header = [str(i) for i in range(784)]
	header.append('label')

	training_data = pd.read_csv(sys.argv[1], names = header)
	training_data.iloc[:,0:784] /= 255

	test_data = pd.read_csv(sys.argv[2], names = header)
	test_data.iloc[:,0:784] /= 255
	return(training_data, test_data)

def dotfunc(a,w,b):
    return (np.dot(w,a) + b)

def kernel_function(y, all_egs, non_zero_gaussian, alpha_gaussian, labels):     # calculates w_T * y
	m = all_egs.shape[0]
	temp = np.sum((all_egs - y)**2, axis = 1)
	temp = scipy.exp((-0.05)*temp).reshape(m,1) * alpha_gaussian.reshape(m,1) * labels.reshape(m,1)
	return (np.sum(temp))    


def one_classifier(training_data, test_data, ind1,ind2,flag):       # if flag = 1, return prediction on whole training_data
    # start1 = datetime.now()

    start = datetime.now()
    training_subset = training_data.loc[training_data['label'].isin([ind1,ind2])]
    training_subset = training_subset.reset_index(drop=True)
    training_subset.label = np.where(training_subset.label == ind1, -1, 1)
    
    test_subset = test_data.loc[test_data['label'].isin([ind1,ind2])]
    test_subset = test_subset.reset_index(drop=True)
    test_subset.label = np.where(test_subset.label == ind1, -1, 1)
    
    # print('read test data and training data ')
    # GAUSSIAN KERNEL
    M = np.array(training_subset.iloc[:,0:784])
    pairwise_dists = squareform(pdist(M))
    K_gaussian_temp = np.exp( -0.05 * pairwise_dists**2)
    yiyj = np.dot( np.array([training_subset.label]).transpose() , np.array([training_subset.label]) )
    K_gaussian = K_gaussian_temp * yiyj
    
    # finding parameters and solving
    m = training_subset.shape[0]
    P = matrix(K_gaussian, tc = 'd')
    q = -1 * matrix(np.ones((m,1)), tc = 'd')
    G = np.zeros((2*m,m))
    h = np.zeros((2*m,1))
    for i in range(m):
        G[i][i] = -1
        G[i+m][i] = 1
        h[i+m] = 1
    G = matrix(G, tc = 'd')
    h = matrix(h, tc = 'd')
    A = matrix(np.array([training_subset['label']]), tc='d')
    B = matrix([0.0])
    sol = solvers.qp(P,q,G,h,A,B)
    alpha_gaussian = np.array(sol['x'])
    
    # support vectors
    non_zero_gaussian = []
    for i in range(len(alpha_gaussian)):
        if (alpha_gaussian[i] > (10**-4)):
            non_zero_gaussian.append(i)
    
    # print("prediction starts ")
    if (flag==0):
        # -------- training set prediction
        labels = np.array([training_subset.label])
        labels = (alpha_gaussian.transpose() * labels)
        predicted = (K_gaussian_temp * labels).sum(axis = 1)

        # calculate b
        svindex = np.argmax(alpha_gaussian>0.0001)
        b_gaussian = training_subset.label[svindex] - predicted[svindex]

        print('b: ' + str(b_gaussian))
        print('model training time: ' + str(datetime.now() - start))

        # prediction continue
        predicted = predicted + b_gaussian
        predicted_training = np.where(predicted<0, ind1, ind2)
        # print(" training set accuracy done ")
        # print(datetime.now() - start1)

        # ---------- test set prediction
        all_egs = np.array(training_subset.iloc[:,0:784])
        labels = training_subset.label
        labels = np.array(labels)
        predicted_test_gaussian = []
        
        test = np.array(test_subset.iloc[:,0:784])
#         for i in range(test_subset.shape[0]):
#             predicted_test_gaussian.append(kernel_function(test[i],all_egs,non_zero_gaussian, alpha_gaussian, labels))

#         for i in range(test_subset.shape[0]):
#             print(i)
#             predicted_test_gaussian.append(kernel_function(test_subset.iloc[i,0:784], all_egs, non_zero_gaussian, alpha_gaussian, labels))
        
        predicted_test_gaussian = np.apply_along_axis(kernel_function, 1, test_subset.iloc[:,0:784], all_egs, non_zero_gaussian, alpha_gaussian, labels)
        predicted_test_gaussian = np.array(predicted_test_gaussian).reshape((test_subset.shape[0],1))
        predicted_test_gaussian = predicted_test_gaussian + b_gaussian
        predicted_test_gaussian = np.where(predicted_test_gaussian < 0, ind1, ind2) 
        
        # print(" test set accuracy done ")
        # print(datetime.now() - start1)
        
    else: #(flag==1):
        # -------- finding b_gaussian
        labels = np.array([training_subset.label])
        labels = (alpha_gaussian.transpose() * labels)
        predicted = (K_gaussian_temp * labels).sum(axis = 1)

        # calculate b
        svindex = np.argmax(alpha_gaussian>0.0001)
        b_gaussian = training_subset.label[svindex] - predicted[svindex]

        # prediction for full data set
        # print('predicting for full training set ')
        all_egs = np.array(training_subset.iloc[:,0:784])
        labels = training_subset.label
        labels = np.array(labels)
        predicted_training = np.apply_along_axis(kernel_function, 1, training_data.iloc[:,0:784], all_egs, non_zero_gaussian, alpha_gaussian, labels)
        predicted_training = predicted_training + b_gaussian
        predicted_training = np.where(predicted_training < 0, ind1, ind2) 
        # print('training set prediction done')

        # ---------- test set prediction
#         all_egs = np.array(training_subset.iloc[:,0:784])
#         labels = training_subset.label
#         labels = np.array(labels)
        predicted_test_gaussian = np.apply_along_axis(kernel_function, 1, test_data.iloc[:,0:784], all_egs, non_zero_gaussian, alpha_gaussian, labels)
        predicted_test_gaussian = predicted_test_gaussian + b_gaussian
        predicted_test_gaussian = np.where(predicted_test_gaussian < 0, ind1, ind2) 
        # print('test set prediction done')
    
    
    predicted_training = np.reshape(predicted_training, len(predicted_training))
    predicted_test_gaussian = np.reshape(predicted_test_gaussian, len(predicted_test_gaussian))
    return(predicted_training, predicted_test_gaussian, non_zero_gaussian, alpha_gaussian)

def partA():
	start = datetime.now()
	training_data, test_data = read_data()
	training_subset = training_data.loc[training_data['label'].isin([4,5])]
	training_subset = training_subset.reset_index(drop=True)
	training_subset.label = np.where(training_subset.label == 4, -1, 1)

	M = np.array(training_subset.iloc[:,0:784])
	K = np.dot(M,M.transpose())
	yiyj = np.dot( np.array([training_subset.label]).transpose() , np.array([training_subset.label]) )
	K = K * yiyj
	# Define QP parameters and solve to find alpha
	m = training_subset.shape[0]
	P = matrix(K, tc = 'd')
	q = -1 * matrix(np.ones((m,1)), tc = 'd')
	G = np.zeros((2*m,m))
	h = np.zeros((2*m,1))
	for i in range(m):
	    G[i][i] = -1
	    G[i+m][i] = 1
	    h[i+m] = 1
	G = matrix(G, tc = 'd')
	h = matrix(h, tc = 'd')
	A = matrix(np.array([training_subset['label']]), tc='d')
	B = matrix([0.0])
	sol = solvers.qp(P,q,G,h,A,B)
	alpha = np.array(sol['x'])
	# number of support vectors
	sv = alpha[np.where(alpha > 0.0001)]
	non_zero_linear = []
	for i in range(len(alpha)):
	    if (alpha[i] > (10** -4)):
	        non_zero_linear.append(i)
	print('nSV: ' + str(len(non_zero_linear)))
	# find w and b
	w = alpha[0] * training_subset.label[0] * np.array(training_subset.iloc[0,0:784])
	for i in non_zero_linear:
	    w += alpha[i] * training_subset.label[i] * np.array(training_subset.iloc[i,0:784])
	svindex = np.argmax(alpha>0.0001)
	b = training_subset.loc[svindex,'label'] - np.dot(w, training_subset.iloc[svindex,0:784])

	print('value of b' + str(b))
	print('model training time: ' + str(datetime.now() - start))

	# training set accuracy 

	predicted = np.apply_along_axis(dotfunc, 1, training_subset.iloc[:,0:784], w, b)
	predicted = np.where(predicted < 0, -1, 1)
	accuracy_array = np.array(training_subset.label) - predicted
	# accuracy_array = accuracy_array.reshape(4000)
	# training_accuracy = 1- (np.count_nonzero(accuracy_array)/len(accuracy_array))
	training_accuracy = (accuracy_array[accuracy_array == 0].shape[0]) / accuracy_array.shape[0]
	print('training accuracy' + str(training_accuracy))

	# test set accuracy
	test_subset = test_data.loc[test_data['label'].isin([4,5])]
	test_subset = test_subset.reset_index(drop=True)
	test_subset.label = np.where(test_subset.label == 4, -1, 1)

	test_predicted = np.apply_along_axis(dotfunc, 1 , test_subset.iloc[:,0:784], w, b)
	test_predicted = np.where(test_predicted<0, -1, 1)
	test_accuracy_array = test_subset.label - test_predicted
	test_accuracy = (test_accuracy_array[test_accuracy_array == 0].shape[0]) / test_accuracy_array.shape[0]
	print('test set accuracy' + str(test_accuracy))

def partB():
	training_data, test_data = read_data()
	predicted_training_g, predicted_test_g, non_zero_g, alpha_g = one_classifier(training_data, test_data, 4,5,0)
	print('nSV: ' + str(len(non_zero_g)))

	print('training set accuracy')
	training_subset = training_data.loc[training_data['label'].isin([4,5])]
	test_subset = test_data.loc[test_data['label'].isin([4,5])]
	    
	accuracy_array_training = training_subset.label - predicted_training_g
	accuracy_g_training = (accuracy_array_training[accuracy_array_training == 0].shape[0]) / accuracy_array_training.shape[0]
	print(accuracy_g_training)

	print('test set accuracy')
	accuracy_array_test = test_subset.label - predicted_test_g
	accuracy_g_test = (accuracy_array_test[accuracy_array_test == 0].shape[0]) / accuracy_array_test.shape[0]
	print(accuracy_g_test)

def partC():
	training_data, test_data = read_data()
	training_subset = training_data.loc[training_data['label'].isin([4,5])]
	test_subset = test_data.loc[test_data['label'].isin([4,5])]
	y  = np.array(training_subset.label)
	x = np.array(training_subset.iloc[:,0:784])
	prob = svm_problem(y,x)

	# LINEAR KERNEL
	print('LINEAR KERNEL: ')
	param = svm_parameter('-t 0 -c 1 -b 1')    ## -t is kernel O(linear) 2(gaussian)
	# print(svm_check_parameter())
	start = datetime.now()
	m = svm_train(prob,param)
	print('linear prediction time ' + str(datetime.now() - start))
	# print('linear kernel model training time' + str(datetime.now() - start))
	# training set - linear kernel 
	p_labs, p_acc, p_vals = svm_predict(y, x, m)
	print('training set accuracy ' + str(p_acc))
	# print(' training accuracy time' + str(datetime.now() - start))

	# test set - linear kernel
	y_test  = np.array(test_subset.label)
	x_test = np.array(test_subset.iloc[:,0:784]) 
	p_labs_test, p_acc_test, p_vals_test = svm_predict(y_test, x_test, m)
	print('test set accuracy ' + str(p_acc_test))
	# print('test accuracy time' + str(datetime.now() - start))

	# GAUSSIAN KERNEL
	print('GAUSSIAN KERNEL: ')
	param = svm_parameter('-g 0.05 -t 2 -c 1 -b 1')    ## -t is kernal O(linear) 2(gaussian)
	start = datetime.now()
	m = svm_train(prob,param)
	print('gaussian training time ' + str(datetime.now() - start))
	# print('model training time ' + str(datetime.now() - start))  
	p_labs, p_acc, p_vals = svm_predict(y, x, m)
	print('training set accuracy ' + str(p_acc))
	# print('model prediction time' + str(datetime.now() - start))
	p_labs_test, p_acc_test, p_vals_test = svm_predict(y_test, x_test, m)
	print('test set accuracy ' + str(p_acc_test))
	# print('model prediction time' + str(datetime.now() - start))

def part2a():
	training_data, test_data = read_data()
	m = training_data.shape[0]
	n = test_data.shape[0]

	train = []
	test = []

	for i in range(10):
	    for j in range(i+1,10):
	        # print('training and predicting ' + str(i) + ' ' + str(j))
	        predicted_training_g, predicted_test_g, non_zero_g, alpha_g = one_classifier(training_data, test_data, i,j,1)
	        train.append(predicted_training_g)
	        test.append(predicted_test_g)
	# print(datetime.now() - start)
	# training set accuracy
	# print('training set')
	actual_labels = training_data.label
	labels_score = [0 for i in range(10)]
	count = 0
	for i in range(m):
	    label_counts = [0 for j in range(10)]
	    for x in train:
	        label_counts[x[i]] += 1
	    predicted_label = label_counts.index(max(label_counts))
	    labels_score[predicted_label] += 1
	    if (predicted_label == actual_labels[i]):
	        count += 1
	train_acc = count/m
	print('training set accuracy' + str(train_acc))

	# test set accuracy
	# print(' test set')
	actual_labels_test = test_data.label
	labels_score_test = [0 for i in range(10)]
	count2 = 0
	for i in range(n):
	    label_counts = [0 for j in range(10)]
	    for x in test:
	        label_counts[x[i]] += 1
	    predicted_label = label_counts.index(max(label_counts))
	    labels_score_test[predicted_label] += 1
	    if (predicted_label == actual_labels_test[i]):
	        count2 += 1
	test_acc = count2/n
	print('test set accuracy' + str(test_acc))

def part2b():
	
	training_data_full, test_data_full = read_data()
	y  = np.array(training_data_full.label)
	x = np.array(training_data_full.iloc[:,0:784])
	start = datetime.now()
	prob = svm_problem(y,x)
	param = svm_parameter('-g 0.05 -t 2 -c 1')    ## -t is kernal O(linear) 2(gaussian)
	
	m = svm_train(prob,param)
	print("time for tarining: " + str(datetime.now() - start))
	# print('model training time' + str(datetime.now() - start))  
	p_labs, p_acc, p_vals = svm_predict(y, x, m)
	print('training set accuracy: ' + str(p_acc))
	y_test  = np.array(test_data_full.label)
	x_test = np.array(test_data_full.iloc[:,0:784])
	p_labs_test, p_acc_test, p_vals_test = svm_predict(y_test, x_test, m)
	print('test set accuracy: ' + str(p_acc_test))
	return (y_test, p_labs_test)

def part2c():
	# CONFUSION MATRIX
	from sklearn.metrics import confusion_matrix
	import matplotlib.pyplot as plt
	import seaborn as sn
	y_test, p_labs_test = part2b()
	
	labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	cm = confusion_matrix(y_test, p_labs_test, labels)  # plabs?
	print(cm)

	df_cm = pd.DataFrame(cm, index=labels, columns = labels)
	plt.figure(figsize = (5,5))
	ax = plt.gca()
	plt.title('CONFUSION MATRIX')
	sn.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
	plt.yticks(rotation=0) 
	ax.hlines([0,1,2,3,4,5,6,7,8,9,10], *ax.get_xlim())
	ax.vlines([0,1,2,3,4,5,6,7,8,9,10], *ax.get_ylim())
	plt.show()

def part2d():
	from sklearn.model_selection import train_test_split

	training_data_full, test_data_full = read_data()
	trainingSet, validationSet = train_test_split(training_data_full, test_size=0.1)

	y  = np.array(trainingSet.label)
	x = np.array(trainingSet.iloc[:,0:784])

	y_validation  = np.array(validationSet.label)
	x_validation = np.array(validationSet.iloc[:,0:784])

	y_test  = np.array(test_data_full.label)
	x_test = np.array(test_data_full.iloc[:,0:784])

	prob = svm_problem(y,x)

	cs = [10**-5 , 10**-3, 1 , 5, 10]
	accuracy_validation = []
	accuracy_test = []
	for cvalue in cs:
	    # print('value of C : ' + str(cvalue))
	    param = svm_parameter('-g 0.05 -t 2')
	    param.C = cvalue
	    m = svm_train(prob,param)
	    # test on validation set
	    p_labs_v, p_acc_v, p_vals_v = svm_predict(y_validation, x_validation, m)
	    accuracy_validation.append(p_acc_v)
	    print('validation set accuracy  ' + str(p_acc_v))
	    # test on test set
	    p_labs_t, p_acc_t, p_vals_t = svm_predict(y_test, x_test, m)
	    accuracy_test.append(p_acc_t)
	    print('test set accuracy  ' + str(p_acc_t))


	# C = [10**-5, 10**-3, 1, 5, 10]
	# accuracy_validation = [8.85, 8.2,97.5, 97.5, 97.5]
	# accuracy_test = [10.1, 9.8, 97.22, 97.34, 97.34]

	X = [-5,-3,0,math.log10(5),1]
	plt.plot(X, accuracy_validation, 'r', label = 'validation accuracy')
	plt.plot(X, accuracy_test, 'b', label = 'test accuracy')
	plt.legend()
	plt.xlabel('log of C')
	plt.ylabel('accuracy')
	plt.show()


if (sys.argv[3] == '1'):
	if (sys.argv[4] == 'a'):
		part2a()
	elif (sys.argv[4] == 'b'):
		y_test, p_labs_test= part2b()
	elif (sys.argv[4] == 'c'):
		part2c()

	else:
		print("hello3")
		part2d()

else:
	if (sys.argv[4] == 'a'):
		partA()
	elif (sys.argv[4] == 'b'):
		partB()
	else:
		partC()




