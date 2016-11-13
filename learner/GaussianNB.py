import numpy as np

from learner.Learner import Learner
from preprocessor.Util import kFold, read_csv, to_labels, compute_accuracy, norm_pdf_univariate

class GaussianNB(Learner):

	"""
	Features independent given class.
	For every distinct class, have mean and variance for all features
	"""
	def __init__(self):
		self.data_by_class = {}
		self.summary_by_class = {}
		self.class_prob = {}

	def seperate_by_class(self, X, y):
		row, features = X.shape
		self.data_by_class = {}
		self.class_prob = {}
		idx = {}
		for i, label in enumerate(y):
			if label not in self.class_prob:
				self.class_prob[label] = 0
			self.class_prob[label] = self.class_prob[label] + 1
		
		for label, freq in self.class_prob.iteritems():
			self.data_by_class[label] = np.zeros((freq, features))
			idx[label] = 0

		for i, entry in enumerate(X):
			label = y[i]
			self.data_by_class[label][idx[label],:] = X[i]
			idx[label] = idx[label] + 1
		return self.data_by_class

	def summarize(self, data):
		return (np.mean(data, axis = 0), np.std(data, axis = 0))
	    
	def set_summary(self):
	    self.summary_by_class	= {}
	    for k,v in self.data_by_class.iteritems():
			self.summary_by_class[k]	= self.summarize(v)

	def add_noise(self, X):
		row, features = X.shape
		np.random.seed(10)
		X += np.random.normal(0, 0.0001, (row, features))
		return X

	def should_add_noise(self, X):
		return np.any(np.std(X, axis = 0)==0)

	def fit(self, X, y):
		rows, features = X.shape
		if self.should_add_noise(X):
			X = self.add_noise(X)
		self.seperate_by_class(X, y)
		self.set_summary()
		for label, instances in self.data_by_class.iteritems():
			self.class_prob[label] = (self.class_prob[label] * 1.0)	/ rows

	def predict(self, X):
		
		rows, num_features = X.shape
		predicted_label = np.zeros(rows)
		
		for i, entry in enumerate(X):
			best_class = -1
			best_prob  = -1e30

			for label, summary in self.summary_by_class.iteritems():
				prob = np.log2(self.class_prob[label])
				for j in range(num_features):
					prob = prob + np.log2(norm_pdf_univariate(entry[j], summary[0][j], summary[1][j]))
				if prob > best_prob:
					best_prob = prob
					best_class = label

			predicted_label[i] = best_class

		return predicted_label