import numpy as np
import pandas as pd
import sys

from preprocessor.Util import kFold, read_csv, to_labels, compute_accuracy

file_name = sys.argv[1]
num_crossval = int(sys.argv[2])

def run_linearDiscriminantAnalysis(X_train, y_train, X_test, y_test):
	from learner.LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
	lda = LinearDiscriminantAnalysis(n_components = 1)
	lda.fit(X_train, y_train)
	return compute_accuracy(y_train, lda.predict(X_train)), compute_accuracy(y_test, lda.predict(X_test))	

def run_sk_linearDiscriminantAnalysis(X_train, y_train, X_test, y_test):
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	lda = LinearDiscriminantAnalysis(n_components = 1)
	lda.fit(X_train, y_train)
	return compute_accuracy(y_train, lda.predict(X_train)), compute_accuracy(y_test, lda.predict(X_test))

def run_linearDiscriminantAnalysis_full(X, y):
	from learner.LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
	lda = LinearDiscriminantAnalysis(n_components = 1)
	lda.fit(X, y)
	return compute_accuracy(y, lda.predict(X))

def run_sk_linearDiscriminantAnalysis_full(X, y):
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	lda = LinearDiscriminantAnalysis(n_components = 1, solver = 'eigen')
	lda.fit(X, y)
	return compute_accuracy(y, lda.predict(X))

print "Reading %s and splitting with 50th percentile" % (file_name) 
X, y = read_csv(file_name)
threshold = np.percentile(y, 50)
y = np.array([1 if entry > threshold else 0 for entry in y])

print "Running on complete data for sanity checks ..."
print "Error using my algorithm ........ %f " %  (1.0 - run_linearDiscriminantAnalysis_full(np.copy(X), np.copy(y)))
print "Error using sklearn algorithm ... %f " % (1.0 - run_sk_linearDiscriminantAnalysis_full(np.copy(X), np.copy(y)))
print "==============================================="
print "KFold validation ..............................."

iteration = 1
accuracy_train = np.zeros(num_crossval)
accuracy_test = np.zeros(num_crossval)

for X_train, X_test, y_train, y_test in kFold(X, y, num_crossval):
	
	train_accuracy, test_accuracy = run_linearDiscriminantAnalysis(np.copy(X_train), np.copy(y_train), np.copy(X_test), np.copy(y_test))
	print "error on training set %f and on test set %f " % (1.0 - train_accuracy, 1.0 - test_accuracy)
	accuracy_train[iteration-1] = 1.0 - train_accuracy
	accuracy_test[iteration-1] = 1.0 - test_accuracy
	iteration = iteration + 1

print "===================================================================="
print "Average error on training data is %f with standard deviation %f " % (np.mean(accuracy_train), np.std(accuracy_train))
print "Average error on test data is %f with standard deviation %f " % (np.mean(accuracy_test), np.std(accuracy_test))
print "===================================================================="
