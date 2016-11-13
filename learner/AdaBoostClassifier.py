import numpy as np

from learner.DecisionTreeClassifier import DecisionTreeClassifier
from learner.Learner import Learner
from learner.Node import Node

class AdaBoostClassifier(Learner):

	def __init__(self,n_estimators = 100):
		self.n_estimators = n_estimators

	def sign(self, csum):
		return -1 if csum < 0 else 1

	def fit(self,X ,y):
		self.X, self.y = X, y
		self.rows, self.features = X.shape
		self.recursive_call = 0
		self.tree_ = np.zeros(self.n_estimators, dtype = Node)
		
		self.weights = np.ones(self.rows) * (1.0 / self.rows)
		self.alphas = np.zeros(self.n_estimators)

		i = 0
		while i < self.n_estimators:
			iter_X, iter_y = self.create_data_set(seed = self.recursive_call)
			self.tree_[i] = DecisionTreeClassifier(random_state = self.recursive_call, should_select_features = False)
			self.tree_[i].fit(iter_X, iter_y)
			self.alphas[i], error = self.get_error(self.tree_[i])
			if error > 0.0 and error < 0.5:
				self.update_weights(self.tree_[i], self.alphas[i])
				i = i + 1
			else:
				self.weights = np.ones(self.rows) * (1.0 / self.rows)
			self.recursive_call = self.recursive_call + 1

	def create_data_set(self, seed = 1):
		np.random.seed(seed)
		idx = np.random.choice(self.rows, self.rows, replace = True, p = self.weights)
		return self.X[idx], self.y[idx]

	def get_error(self, clf):
		error = 0.0
		"""
		Making sure that error is calculated on original data-set not bootstrapped data-set.
		"""
		for i, entry in enumerate(self.X):
			error = error + self.weights[i] * (1.0 if clf.predict(entry) != self.y[i] else 0)
		alpha = 0.5 * np.log((1.0 - error) / error)
		return alpha, error

	def update_weights(self, clf, alpha):
		"""
		Updating weights of all examples from training data.
		"""
		for i, entry in enumerate(self.X):
			self.weights[i] = self.weights[i] * np.exp(-1.0 * alpha * self.y[i] * clf.predict(entry))
		self.weights = self.weights / np.sum(self.weights)

	def predict(self, X):
		result = []
		for entry in X:
			csum = 0.0
			for i in range(self.n_estimators):
				csum = csum + self.alphas[i] * self.tree_[i].predict(entry)			
			result.append(self.sign(csum))
		return result