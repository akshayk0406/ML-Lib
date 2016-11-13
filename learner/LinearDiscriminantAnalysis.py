import numpy as np

from numpy.linalg import inv
from learner.Learner import Learner
from preprocessor.Util import norm_pdf_multivariate

class LinearDiscriminantAnalysis(Learner):

	def __init__(self, n_components = 1):
		self.params_ = []
		self.n_components = n_components
		self.threshold = 0.0

	def separate_by_class(self, X, y):
		row, features = X.shape
		data_by_class = {}
		class_freq = {}
		idx = {}
		for i, label in enumerate(y):
			if label not in class_freq:
				class_freq[label] = 0
			class_freq[label] = class_freq[label] + 1
		
		for label, freq in class_freq.iteritems():
			data_by_class[label] = np.zeros((freq, features))
			idx[label] = 0

		for i, entry in enumerate(X):
			label = y[i]
			data_by_class[label][idx[label],:] = X[i]
			idx[label] = idx[label] + 1
		return data_by_class

	def get_mean_by_class(self, data_by_class):
		mean_by_class = {}
		for label, value in data_by_class.iteritems():
			mean_by_class[label] = np.mean(value, axis = 0)
		return mean_by_class

	def get_cov_by_class(self, data_by_class, num_features):
		cov_by_class = {}
		for label, value in data_by_class.iteritems():
			cov_by_class[label] = np.cov(value.T)
		return cov_by_class

	def within_class_variance(self, data_by_class, mean_by_class, num_features):
		var_within_class = np.zeros((num_features, num_features))
		for label, entry in data_by_class.iteritems():
			scatter_matrix = np.zeros((num_features, num_features))
			for row in entry:
				row, mean = row.reshape(num_features, 1), mean_by_class[label].reshape(num_features, 1)
				scatter_matrix += (row - mean).dot((row-mean).T)
			var_within_class = var_within_class + scatter_matrix
		return var_within_class

	def between_class_variance(self, data_by_class, mean_by_class, overall_mean, num_features):
		var_between_class = np.zeros((num_features, num_features))
		for label, entry in mean_by_class.iteritems():
			entry = entry.reshape(num_features, 1)
			overall_mean = overall_mean.reshape(num_features, 1)
			var_between_class += len(data_by_class[label]) * (entry - overall_mean).dot((entry - overall_mean).T)
		return var_between_class

	def compute_class_priors(self, y):
		instances = 0
		class_freq = {}
		for entry in y:
			if entry not in class_freq:
				class_freq[entry] = 0
			class_freq[entry] = class_freq[entry] + 1
			instances = instances + 1
		for label, frequency in class_freq.iteritems():
			class_freq[label] = (class_freq[label] * 1.0) / (instances * 1.0)
		return class_freq
	
	def fit(self, X, y):
		rows, features = X.shape
		data_by_class = self.separate_by_class(X, y)
		mean_by_class = self.get_mean_by_class(data_by_class)
		self.class_priors_ = self.compute_class_priors(y)

		var_within_class = self.within_class_variance(data_by_class, mean_by_class, features)
		var_between_class = self.between_class_variance(data_by_class, mean_by_class, np.mean(X, axis = 0), features)
		
		if not np.isfinite(np.linalg.cond(var_within_class)):
			var_within_class = np.linalg.pinv(var_within_class)
		else:
			var_within_class = np.linalg.inv(var_within_class)


		eigen_values, eigen_vectors = np.linalg.eig(var_within_class.dot(var_between_class))
		# sort eigen_values
		eigen_value_pairs = []
		for i in range(len(eigen_values)):
			eigen_value_pairs.append((np.abs(eigen_values[i]), eigen_vectors[:, i]))
		eigen_value_pairs = sorted(eigen_value_pairs, key = lambda k: k[0], reverse = True)
		self.params_ = np.zeros((features, self.n_components))
		for i in range(len(eigen_values)):
			for j in range(self.n_components):
				self.params_[i][j] = eigen_value_pairs[j][1][i].real
		self.X_lda = X.dot(self.params_)
		self.data_by_class_transformed = self.separate_by_class(self.X_lda, y)
		self.mean_by_class_transformed = self.get_mean_by_class(self.data_by_class_transformed)
		self.cov_by_class = self.get_cov_by_class(self.data_by_class_transformed, features)
		if 1 == self.n_components:
			self.threshold = np.mean(self.mean_by_class_transformed.values(), axis = 0)

	def transform(self, X):
		return X.dot(self.params_)

	def fit_transform(self, X, y):
		self.fit(X, y)
		return self.transform(X)

	def predict(self, X):
		row, features = X.shape
		predictions = np.zeros(row)
		X = X.dot(self.params_)
		if 1 == self.n_components:
			return np.array(map(lambda x: 0 if x > self.threshold else 1, X))		

		for i, entry in enumerate(X):
			best_prob = 0.0
			best_label = -1
			for label, class_mean in self.mean_by_class_transformed.iteritems():
				cur_prob = norm_pdf_multivariate(entry, class_mean, self.cov_by_class[label])
				cur_prob = cur_prob * self.class_priors_[label]
				if cur_prob > best_prob:
					best_prob = cur_prob
					best_label = label
			predictions[i] = best_label
		return predictions