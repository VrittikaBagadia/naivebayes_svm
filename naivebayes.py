import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import json
from datetime import datetime
import math
import random
from sklearn.metrics import confusion_matrix
import sklearn
import sys
# import matplotlib.pyplot as plt
# import numpy as np

# import pandas as pd
# import seaborn as sn

# PART A
def partA():
	training_data = []
	vocabulary = {}
	vocabulary_size = 0
	star_examples = [0,0,0,0,0]

	li = []
	for i in range(5):
		li.append([])
	with open(sys.argv[1]) as f:
	    for line in f:
	    	temp = json.loads(line)
	    	rating = int(temp['stars'])
	    	# review = tokenizer.tokenize(temp['text'])
	    	review = CountVectorizer().build_analyzer()(temp['text'])
	    	training_data.append((rating,review))
	    	star_examples[rating-1] += 1
	    	for token in review:
	            
	    		if token not in vocabulary.keys():
	    			vocabulary[token] = vocabulary_size
	    			vocabulary_size += 1
	    			for i in range(5):
	    				li[i].append(0)
	    		index = vocabulary[token]
	    		li[rating-1][index] += 1
	f.close()

	parameters = []
	for i in range(5):
		l = []
		sumi = sum(li[i])
		for k in range(vocabulary_size):
			phi_k_given_i = math.log((li[i][k] + 1)/(sumi + vocabulary_size))
			l.append(phi_k_given_i)
		parameters.append(l)
	    
	# calculating class probabilities
	class_prob = []
	m = sum(star_examples)
	for i in range(5):
		class_prob.append(star_examples[i]/m)

	# computing accuracy on training set
	correct = 0
	total = 0
	for example in training_data:
	    total += 1
	    p1 = math.log(class_prob[0])
	    p2 = math.log(class_prob[1])
	    p3 = math.log(class_prob[2])
	    p4 = math.log(class_prob[3])
	    p5 = math.log(class_prob[4])
	    for token in example[1]:
	        index = vocabulary[token]
	        p1 += parameters[0][index]
	        p2 += parameters[1][index]
	        p3 += parameters[2][index]
	        p4 += parameters[3][index]
	        p5 += parameters[4][index]
	    probs = [p1,p2,p3,p4,p5]
	    calc_class = probs.index(max(probs)) + 1
	    rating = example[0]
	    if (calc_class == rating):
	        correct += 1;
	print(str(correct/total * 100) + " percent accuracy on training set")

	# calculating accuracy on test set
	correct = 0
	total = 0
	classes_input = []
	classes_predicted = []
	test_data = []
	with open(sys.argv[2]) as f:
	    for line in f:
	        total+=1
	        temp = json.loads(line)
	        rating = int(temp['stars'])
	        review = CountVectorizer().build_analyzer()(temp['text'])
	        test_data.append((rating, review))
	        probs = []
	        for i in range(5):
	            probs.append(math.log(class_prob[i]))
	        for token in review:
	            try:
	                index = vocabulary[token]
	            except:
	                continue
	            for i in range(5):
	                probs[i] += parameters[i][index]

	        calc_class = probs.index(max(probs)) + 1
	        classes_input.append(rating)
	        classes_predicted.append(calc_class)
	        if (calc_class == rating):
	            correct += 1;
	f.close()
	nb_accuracy = correct/total * 100
	print(str(nb_accuracy) + " percent accuracy on test set")

	return(classes_input, classes_predicted)

# PART B
def partB():
	correct_random = 0
	correct_majority = 0
	total = 0
	class_totals = []
	training_ratings = [0 for i in range(5)]
	with open(sys.argv[1]) as f:
	    for line in f:
	        temp = json.loads(line)
	        rating = int(temp['stars'])
	        training_ratings[rating-1] += 1
	f.close()
	majority_rating = training_ratings.index(max(training_ratings)) + 1
	for i in range(5):
	    class_totals.append(0)
	with open(sys.argv[2]) as f:
	    for line in f:
	        total += 1
	        temp = json.loads(line)
	        rating = int(temp['stars'])
	        class_totals[rating-1] += 1
	        random_pred = random.randint(1,5)
	        if (rating == random_pred):
	            correct_random += 1
	        if (rating == majority_rating):
	            correct_majority += 1
	random_accuracy = correct_random / total * 100
	majority_accuracy = correct_majority / total * 100
	# majority_accuracy = max(class_totals) / total * 100
	print(str(random_accuracy) + ' percent accuracy in case of random prediction ')
	print(str(majority_accuracy) + ' percent accuracy in case of majority prediction ')
	# print('improvement:')
	# print('random: ' + str((nb_accuracy - random_accuracy) / random_accuracy * 100))
	# print('majority:' + str((nb_accuracy - majority_accuracy) / majority_accuracy * 100))

# PART C
def partC():
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import seaborn as sn
	classes_input, classes_predicted = partA()
	labels = [1, 2, 3, 4, 5]
	cm = confusion_matrix(classes_input, classes_predicted, labels)
	print(cm)
	df_cm = pd.DataFrame(cm, index=labels, columns = labels)
	plt.figure(figsize = (5,5))
	ax = plt.gca()
	plt.title('CONFUSION MATRIX')
	sn.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
	plt.yticks(rotation=0) 
	ax.hlines([0,1,2,3,4,5], *ax.get_xlim())
	ax.vlines([0,1,2,3,4,5], *ax.get_ylim())
	plt.show()

def partD():
	nltk.download('stopwords')
	from nltk.corpus import stopwords
	from nltk.stem.porter import PorterStemmer
	stoplist = set(stopwords.words('english'))
	porter = PorterStemmer()
	stemmed_training_data = []
	star_examples = [0,0,0,0,0]
	with open(sys.argv[1]) as f:
	    for line in f:
	    	temp = json.loads(line)
	    	rating = int(temp['stars'])
	    	# review = tokenizer.tokenize(temp['text'])
	    	review = CountVectorizer().build_analyzer()(temp['text'])
	    	review = [porter.stem(word) for word in review if word not in stoplist]
	    	stemmed_training_data.append((rating,review))
	    	star_examples[rating-1] += 1
	  
	# 1. vocabulary
	stemmed_vocabulary = {}
	stemmed_vocabulary_size = 0
	
	counter = 0
	occurrences = []
	for i in range(5):
	    occurrences.append([])
	for example in stemmed_training_data:
	    rating = example[0]
	    review = example[1]
	    for token in review:
	        if token not in stemmed_vocabulary.keys():
	            stemmed_vocabulary[token] = stemmed_vocabulary_size
	            stemmed_vocabulary_size += 1
	            for i in range(5):
	                occurrences[i].append(0)
	        index = stemmed_vocabulary[token]
	        occurrences[rating-1][index] += 1
	        
	# parameter calculation
	stemmed_parameters = []
	for i in range(5):
	    l = []
	    sumi = sum(occurrences[i])
	    for k in range(stemmed_vocabulary_size):
	        phi_k_given_i = math.log((occurrences[i][k] + 1)/(sumi + stemmed_vocabulary_size))
	        l.append(phi_k_given_i)
	    stemmed_parameters.append(l)

	# calculating class probabilities
	class_prob = []
	m = sum(star_examples)
	for i in range(5):
		class_prob.append(star_examples[i]/m)

	# computing accuracy on training set
	correct = 0
	total = 0
	for example in stemmed_training_data:
	    total += 1
	    rating = example[0]
	    review = example[1]
	    p1 = math.log(class_prob[0])
	    p2 = math.log(class_prob[1])
	    p3 = math.log(class_prob[2])
	    p4 = math.log(class_prob[3])
	    p5 = math.log(class_prob[4])
	    for token in review:
	        index = stemmed_vocabulary[token]
	        p1 += stemmed_parameters[0][index]
	        p2 += stemmed_parameters[1][index]
	        p3 += stemmed_parameters[2][index]
	        p4 += stemmed_parameters[3][index]
	        p5 += stemmed_parameters[4][index]
	    probs = [p1,p2,p3,p4,p5]
	    calc_class = probs.index(max(probs)) + 1
	    if (calc_class == rating):
	        correct += 1;
	print(str(correct/total * 100) + " percent accuracy on training set")

	# calculating accuracy on test set
	correct = 0
	total = 0
	# classes_input = []
	stemmed_classes_predicted = []
	# stemmed_test_data = []
	with open(sys.argv[2]) as f:
	    for line in f:
	        temp = json.loads(line)
	        rating = int(temp['stars'])
	        review = CountVectorizer().build_analyzer()(temp['text'])
	        review = [porter.stem(word) for word in review if word not in stoplist]
	        # stemmed_test_data.append((rating, review))
	        total+=1
	    
	        probs = []
	        for i in range(5):
	            probs.append(math.log(class_prob[i]))
	        for token in review:
	            try:
	                index = stemmed_vocabulary[token]
	            except:
	                continue
	            for i in range(5):
	                probs[i] += stemmed_parameters[i][index]
	        calc_class = probs.index(max(probs)) + 1
	        stemmed_classes_predicted.append(calc_class)
	        if (calc_class == rating):
	            correct += 1;
	f.close()
	stemmed_nb_accuracy = correct/total * 100
	print(str(stemmed_nb_accuracy) + ' percent accuracy on test set with stop word removal and stemming')
	
def partE():
	# using bigram + POS tagging
	# vocabulary building
	import nltk
	nltk.download('averaged_perceptron_tagger')
	training_data = []
	vocabulary = {}
	vocabulary_size = 0
	star_examples = [0,0,0,0,0]
	bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5)
	analyze = bigram_vectorizer.build_analyzer()
	li = []
	for i in range(5):
	    li.append([])
	with open(sys.argv[1]) as f:
	    for line in f:
	        temp = json.loads(line)
	        rating = int(temp['stars'])
	        review = nltk.pos_tag(analyze(temp['text']))
	        review = [i[0]+i[1] for i in review]
	        training_data.append((rating,review))
	        star_examples[rating-1] += 1
	        for token in review: 
	            if token not in vocabulary.keys():
	                vocabulary[token] = vocabulary_size
	                vocabulary_size += 1
	                for i in range(5):
	                    li[i].append(0)
	            index = vocabulary[token]
	            li[rating-1][index] += 1
	f.close()

	# calculating probability phi_k given y
	parameters = []
	for i in range(5):
		l = []
		sumi = sum(li[i])
		for k in range(vocabulary_size):
			phi_k_given_i = math.log((li[i][k] + 1)/(sumi + vocabulary_size))
			l.append(phi_k_given_i)
		parameters.append(l)
	    
	# calculating class probabilities
	class_prob = []
	m = sum(star_examples)
	for i in range(5):
		class_prob.append(star_examples[i]/m)

	# computing accuracy on training set
	# start = datetime.now()
	correct = 0
	total = 0
	for example in training_data:
	    total += 1
	    rating = example[0]
	    review = example[1]
	    p1 = math.log(class_prob[0])
	    p2 = math.log(class_prob[1])
	    p3 = math.log(class_prob[2])
	    p4 = math.log(class_prob[3])
	    p5 = math.log(class_prob[4])
	    for token in review:
	        index = vocabulary[token]
	        p1 += parameters[0][index]
	        p2 += parameters[1][index]
	        p3 += parameters[2][index]
	        p4 += parameters[3][index]
	        p5 += parameters[4][index]
	    probs = [p1,p2,p3,p4,p5]
	    calc_class = probs.index(max(probs)) + 1
	    if (calc_class == rating):
	        correct += 1;
	print(str(correct/total * 100) + " percent accuracy on training set")

	# calculating accuracy on test set
	correct = 0
	total = 0
	classes_input = []
	classes_predicted = []
	test_data = []
	with open(sys.argv[2]) as f:
	    for line in f:
	        total+=1
	        temp = json.loads(line)
	        rating = int(temp['stars'])
	        # review = analyze(temp['text'])
	        review = nltk.pos_tag(analyze(temp['text']))
	        review = [i[0]+i[1] for i in review]
	        test_data.append((rating, review))
	        probs = []
	        for i in range(5):
	            probs.append(math.log(class_prob[i]))
	        for token in review:
	            try:
	                index = vocabulary[token]
	            except:
	                continue
	            for i in range(5):
	                probs[i] += parameters[i][index]

	        calc_class = probs.index(max(probs)) + 1
	        classes_input.append(rating)
	        classes_predicted.append(calc_class)
	        if (calc_class == rating):
	            correct += 1;
	nb_accuracy = correct/total * 100
	print(str(nb_accuracy) + " percent accuracy on test set")
	f.close()
	return(classes_input, classes_predicted)

def partF():
	classes_input, classes_predicted = partE()
	print(sklearn.metrics.classification_report(classes_input, classes_predicted))
	print(sklearn.metrics.confusion_matrix(classes_input, classes_predicted))


if (sys.argv[3] == 'a'):
	classes_input, classes_predicted = partA()
elif (sys.argv[3] == 'b'):
	partB()
elif (sys.argv[3] == 'c'):
	partC()
elif (sys.argv[3] == 'd'):
	partD()
elif (sys.argv[3] == 'e'):
	classes_input, classes_predicted = partE()

elif (sys.argv[3] == 'g'):
	print('hi')