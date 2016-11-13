import matplotlib.pyplot as plt
import numpy as np
import sys

from learner.GaussianNB import GaussianNB
from preprocessor.Util import kFold, read_csv, to_labels, compute_accuracy, stratified_split, get_file_name, is_boston_dataset

def run_nb(X_train, y_train, X_test, y_test):
	nb = GaussianNB()
	nb.fit(X_train, y_train)
	return compute_accuracy(y_test, nb.predict(X_test))

def run(file_name, num_splits, split_ratio, X, y):
	nb_errors = np.zeros((num_splits, len(split_ratio)))

	for i in range(num_splits):
		print "Running iteration %d" % (i+1)
		X_train, X_test, y_train, y_test = stratified_split(X, y, i)
		y_train = np.array([int(label) for label in y_train])
		y_test = np.array([int(label) for label in y_test])
		tr_row, features = X_train.shape
		for j in range(len(split_ratio)):
			required_size = int(tr_row * split_ratio[j])
			nb_errors[i][j] = 1.0 - run_nb(np.copy(X_train[0:required_size,:]), np.copy(y_train[0:required_size]), np.copy(X_test), np.copy(y_test))

	nb_error_mean = np.mean(nb_errors, axis = 0)
	nb_error_std =  np.std(nb_errors, axis = 0)
	
	print "Error using Naive Bayes on different splits ...."
	print nb_error_mean
	print "===================================================="

file_name = sys.argv[1]
num_splits = int(sys.argv[2])
split_ratio = [float(int(x)*1.0/ 100.0) for x in sys.argv[3].split(',')]

print "Reading %s" % (file_name) 
X, y = read_csv(file_name)
if is_boston_dataset(file_name):
	print "Running on Boston50 ..... "
	run(get_file_name(file_name) + '_50.png' , num_splits, split_ratio, X, to_labels(y, 50))
	print "Running on Boston75 ..... "
	X, y = read_csv(file_name)
	run(get_file_name(file_name) + '_75.png' , num_splits, split_ratio, X, to_labels(y, 75))
else:
	run(get_file_name(file_name), num_splits, split_ratio, X, y)
