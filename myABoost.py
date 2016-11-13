import matplotlib.pyplot as plt
import numpy as np
import sys

from learner.AdaBoostClassifier import AdaBoostClassifier
from preprocessor.Util import kFold, read_csv, to_labels

def plot_learning_curve(training_error_mean, training_error_std, testing_error_mean, testing_error_std, num_features, file_name):
    
    plt.figure()
    plt.title("Training and Testing accuracy as number of estimators changes")

    plt.xlabel("Number of Estimators (B)")
    plt.ylabel("Error")
    plt.grid()

    plt.fill_between(num_features, training_error_mean - training_error_std, training_error_mean + training_error_std, alpha=0.1, color="r")
    plt.fill_between(num_features, testing_error_mean - testing_error_std, testing_error_mean + testing_error_std, alpha=0.1, color="g")
    plt.plot(num_features, training_error_mean, 'o-', color="r", label="Training Error")
    plt.plot(num_features, testing_error_mean, 'o-', color="g", label="Testing Error")
    plt.legend(loc="best")
    plt.savefig(file_name)

def run_AdaBoost(num_base_classifier, X_train, X_test, y_train, y_test):
	clf = AdaBoostClassifier(n_estimators = num_base_classifier)
	clf.fit(X_train, y_train)
	return np.mean(y_train == clf.predict(X_train)), np.mean(y_test == clf.predict(X_test))

def run_AdaBoost_sklearn(num_base_classifier, X_train, X_test, y_train, y_test):
	from sklearn.ensemble import AdaBoostClassifier
	clf = AdaBoostClassifier(n_estimators = num_base_classifier)
	clf.fit(X_train, y_train)
	return np.mean(y_train == clf.predict(X_train)), np.mean(y_test == clf.predict(X_test))

def run(X, y, B, cross_val):

	y = np.array(map(lambda x: 1 if x == 1 else -1, y))
	tr_accuracy, te_accuracy, tr_std, te_std = np.zeros(len(B)), np.zeros(len(B)), np.zeros(len(B)), np.zeros(len(B))
	for i, num_base_classifier in enumerate(B):
		num_iteration = 0
		performance_tr, performance_te = np.zeros(cross_val), np.zeros(cross_val)
		for X_train, X_test, y_train, y_test in kFold(X, y, cross_val):
			train_accuracy, test_accuracy = run_AdaBoost(num_base_classifier, X_train, X_test, y_train, y_test)
			performance_tr[num_iteration] = 1.0 - train_accuracy
			performance_te[num_iteration] = 1.0 - test_accuracy
			num_iteration = num_iteration + 1
		tr_accuracy[i], tr_std[i] = np.mean(performance_tr), np.std(performance_tr)
		te_accuracy[i], te_std[i] = np.mean(performance_te), np.std(performance_te)
		print "Analysis when number of base classifiers is %d " % (num_base_classifier)
		print "Error on Training set"
		print performance_tr
		print "Error on test set"
		print performance_te
		print "Average Error on training set is %f with std deviation %f " % (np.mean(performance_tr), np.std(performance_tr))
		print "Average Error on test set is %f with std deviation %f " % (np.mean(performance_te), np.std(performance_te))
		print " ===================================================== \n"
	return tr_accuracy, tr_std, te_accuracy, te_std

file_name = sys.argv[1]
B = [int(x.strip()) for x in sys.argv[2].split(',')]
cross_val = int(sys.argv[3])

print "Reading %s" % (file_name)
X, y = read_csv(file_name)
print "Running on Boston50 ..."
tr_accuracy, tr_std, te_accuracy, te_std = run(X, to_labels(y, 50), B, cross_val)
#plot_learning_curve(tr_accuracy, tr_std, te_accuracy, te_std, B, 'AdaBoost_Boston50.png')
print " =================== "
print "Running on Boston75 ..."
tr_accuracy, tr_std, te_accuracy, te_std = run(X, to_labels(y, 75), B, cross_val)
#plot_learning_curve(tr_accuracy, tr_std, te_accuracy, te_std, B, 'AdaBoost_Boston75.png')