import numpy as np

from learner.DecisionTreeClassifier import DecisionTreeClassifier
from learner.Learner import Learner
from learner.Node import Node
import time

class RandomForestClassifier(Learner):

	def __init__(self, max_depth = 2, num_features = 13, n_estimators = 100):
		self.max_depth = max_depth
		self.num_features = num_features
		self.n_estimators = n_estimators

	def fit(self,X ,y):
		self.X, self.y = X, y
		self.rows, self.features = X.shape
		self.recursive_call = 0
		self.tree_ = np.zeros(self.n_estimators, dtype = Node)
		for i in range(self.n_estimators):
			iter_X, iter_y = self.create_data_set(seed = self.recursive_call)
			self.tree_[i] = DecisionTreeClassifier(random_state = self.recursive_call,
							num_features = self.num_features,
							should_select_features = True)
			self.tree_[i].fit(iter_X, iter_y)
			self.recursive_call = self.recursive_call + 1

	def predict(self, X):
		result = []
		for j, entry in enumerate(X):
			predictions = np.zeros(self.n_estimators)
			for i in range(self.n_estimators):
				predictions[i] = self.tree_[i].predict(entry)
			labels, counts = np.unique(predictions, return_counts = True)
			result.append(labels[np.argmax(counts)])
		return result

	def create_data_set(self, seed = 1):
		np.random.seed(seed)
		idx = np.random.choice(self.rows, self.rows, replace = True)
		return self.X[idx], self.y[idx]