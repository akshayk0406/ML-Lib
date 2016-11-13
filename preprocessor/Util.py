import math
import numpy as np
import pandas as pd

from numpy.linalg import inv
from scipy import linalg

def train_test_split(X, y, test_size = 0.3):
	num_elements = len(X)
	test_size = int(test_size * num_elements)
	train_size = num_elements - test_size
	X_train, y_train = X[:train_size], X[:train_size]
	X_test, y_test = X[train_size+1:], X[train_size+1:]
	return X_train, X_test, y_train, y_test

def shuffle(X, y, seed = 12):
	
	num_elements, features = X.shape
	intermediate_df = np.hstack((X, y.reshape(num_elements,1)))
	np.random.seed(seed)
	np.random.shuffle(intermediate_df)
	y = intermediate_df[:, features]
	X = np.delete(intermediate_df, -1, 1)
	return X, y
	
def kFold(X, y, k = 10):
	
	X,y = shuffle(X, y)
	num_elements, features = X.shape
	size_per_fold = num_elements / k
	
	for i in range(k):
		if 0 == i:
			yield np.copy(X[size_per_fold:]), np.copy(X[0:size_per_fold]), np.copy(y[size_per_fold:]), np.copy(y[0:size_per_fold])
		else:
			start = i * size_per_fold
			end = (i+1) * size_per_fold
			yield np.copy(np.vstack((X[:start], X[end:]))), np.copy(X[start:end]), np.copy(np.append(y[:start], y[end:])), np.copy(y[start:end])

def stratified_split(X, y, seed, train_size_ratio = 0.8):
	
	X, y = shuffle(X, y, seed)
	num_elements, features = X.shape

	class_freq = {}
	cur_class_freq = {}
	for i, label in enumerate(y):
		if label not in class_freq:
			class_freq[label] = 0
			cur_class_freq[label] = 0
		class_freq[label] = class_freq[label] + 1

	train_size = 0
	for label, freq in class_freq.iteritems():
		class_freq[label] = int(class_freq[label] * train_size_ratio)
		train_size = train_size + class_freq[label]
	test_size = num_elements - train_size
	
	X_train = np.zeros((train_size, features))
	X_test = np.zeros((test_size, features))
	y_train = np.zeros(train_size)
	y_test = np.zeros(test_size)
	cur_train_idx = 0
	cur_test_idx = 0

	for i, entry in enumerate(X):
		label = y[i]
		if cur_class_freq[label] < class_freq[label]:
			X_train[cur_train_idx,:] = X[i]
			y_train[cur_train_idx] = label
			cur_class_freq[label] = cur_class_freq[label] + 1
			cur_train_idx = cur_train_idx + 1
		else:
			X_test[cur_test_idx,:] = X[i]
			y_test[cur_test_idx] = label
			cur_test_idx = cur_test_idx + 1

	return np.copy(X_train), np.copy(X_test), np.copy(y_train), np.copy(y_test)


def read_csv(file_name):
	data = pd.read_csv(file_name, dtype = 'float64')
	total_columns = len(data.columns.tolist())
	y = data.iloc[:,total_columns-1]
	X = data.drop(data.columns[[total_columns-1]], axis=1)
	return X.as_matrix(), np.array(y)

def to_labels(inp, percentile):
	threshold = np.percentile(inp, percentile)
	return np.array([1 if x > threshold else 0 for x in inp])

def compute_accuracy(given, predicted):
	instances = 0
	correct = 0
	for i in range(len(given)):
		if given[i] == predicted[i]:
			correct = correct + 1
		instances = instances + 1
	return (correct*1.0) / (instances*1.0)

def get_file_name(file_name):
	tokens = file_name.split('/')
	if len(tokens) > 0:
		name = tokens[len(tokens)-1]
		name_tokens = name.split('.')
		if len(name_tokens) > 0:
			return name_tokens[0]
	return 'test'

def is_boston_dataset(file_name):
	return get_file_name(file_name).find('boston') >= 0

def norm_pdf_multivariate(x, mu, sigma):
	norm_const = 1.0/(math.pow((2* math.pi),float(len(x))/2) * math.pow(linalg.det(sigma),1.0/2))
	result = math.pow(math.e, -0.5 * (np.matrix(x - mu) * np.linalg.inv(sigma) * np.matrix(x - mu).T))
	return norm_const * result

def norm_pdf_univariate(x, mu, sd):
	if 0.0 == sd:
		return 1.0 if x == mu else 1e-30
	var = float(sd)**2
	denom = (2 * math.pi* var)**.5
	num = math.exp(-(float(x)-float(mu))**2/(2*var))
	return num/denom
