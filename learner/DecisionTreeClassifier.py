import numpy as np

from learner.Learner import Learner
from learner.Node import Node

class DecisionTreeClassifier(Learner):

	def __init__(self, max_depth = 2, num_features = 13, random_state = 0, should_select_features = False):
		self.max_depth = max_depth
		self.num_features = num_features
		self.recursive_call = random_state
		self.should_select_features = should_select_features

	def fit(self,X ,y):
		self.X, self.y = X, y
		self.rows, self.features = X.shape
		self.tree_ = self._construct(X, y, 0)
		
	def predict(self, X):
		return self._predict(self.tree_, X)

	def _predict(self, tree, entry):
		if tree.isLeaf:
			return tree.label
		else:
			if entry[tree.attr] <= tree.threshold:
				return self._predict(tree.left, entry) if tree.left else self._predict(tree.right, entry)
			else:
				return self._predict(tree.right, entry) if tree.right else self._predict(tree.left, entry)

	def split_points(self, X, feature_idx):
		result = []
		required_percentiles = [10, 20, 30, 50, 60, 70, 80, 90]
		for percentile in required_percentiles:
			result.append(np.percentile(X[:,feature_idx], percentile))
		return result

	def compute_entropy(self, y):
		freq = np.unique(y, return_counts = True)
		n = len(y)
		csum = 0.0
		for key, ct in zip(*freq):
			csum = csum + (ct*1.0 / n) * np.log2(ct*1.0 / n)
		return -1.0 * csum

	def is_binary(self, X, feature_idx):
		return len(np.unique(X[:,feature_idx])) == 2

	def is_pure(self, y):
		return len(np.unique(y)) == 1

	def _computeOptimalSplit(self, X, y):
		np.random.seed(self.recursive_call)
		if self.should_select_features:
			feasible_features = np.random.choice(self.features, self.num_features, replace = False)
		else:
			feasible_features = np.arange(self.features)
		best_attribute, threshold, min_entropy = None, None, 1e30
		for feature_idx in feasible_features:
			values = np.unique(X[:,feature_idx])
			if len(values) == 2:
				values.sort()
				left_idx, right_idx = X[:,feature_idx] == values[0], X[:, feature_idx] == values[1]
				y_left, y_right = y[left_idx], y[right_idx]
				entropy = (len(y_left) * 1.0 / len(y)) * self.compute_entropy(y_left)
				entropy = entropy +  (len(y_right) * 1.0 / len(y)) * self.compute_entropy(y_right)
				if entropy < min_entropy:
					min_entropy = entropy
					best_attribute = feature_idx
					threshold = min(values[0], values[1])
			else:
				values = self.split_points(X, feature_idx)
				for cur_threshold in values:
					left_idx, right_idx = X[:,feature_idx] <= cur_threshold, X[:,feature_idx] > cur_threshold
					y_left, y_right = y[left_idx], y[right_idx]
					entropy = (len(y_left) * 1.0 / len(y)) * self.compute_entropy(y_left)
					entropy = entropy +  (len(y_right) * 1.0 / len(y)) * self.compute_entropy(y_right)
					if entropy < min_entropy:
						min_entropy = entropy
						best_attribute = feature_idx
						threshold = cur_threshold

		return best_attribute, threshold, min_entropy

	def _construct(self, X, y, cur_depth):

		self.recursive_call = self.recursive_call  + 1
		(values,counts) = np.unique(y ,return_counts=True)
		if len(values) == 1 or cur_depth == self.max_depth:
			return Node(-1, -1, True, values[np.argmax(counts)])
		else:
			attr, thresh, min_entropy = self._computeOptimalSplit(X, y)
			cur_node = Node(attr, thresh)
			left_idx, right_idx = X[:,attr] <= thresh, X[:,attr] > thresh
			if np.sum(left_idx == True) > 0:
				X_left, y_left = X[left_idx], y[left_idx]
				cur_node.left = self._construct(X_left, y_left, cur_depth + 1)
			if np.sum(right_idx == True) > 0:
				X_right, y_right = X[right_idx], y[right_idx]
				cur_node.right = self._construct(X_right, y_right, cur_depth + 1)
			return cur_node