import numpy as np
from learner.Learner import Learner
from preprocessor.StandardScaler import StandardScaler

class LogisticRegression(Learner):

	def __init__(self):
		self.learning_rate = 0.01
		self.iterations = 5000
		self.threshold = 0.0001

	def sigmoid(self, X):
		return 1.0 / (1.0 + np.exp(-X))

	def cost(self, X, y, label):
		c1 = y * (self.theta_[label].dot(X.T))
		c2 = np.log2(1.0 + self.sigmoid(self.theta_[label].dot(X.T)))
		return (-1.0 * np.sum(c1 - c2)) / self.row

	def should_normalize(self, X):
		return len(np.where(np.any(np.std(X, axis = 0)==0))[0]) == 0

	def gradient(self, X, y, label):
		return ((self.sigmoid(self.theta_[label].dot(X.T)) - y).dot(X)) / self.row

	def train(self, X, y, label):
		for i in range(self.iterations):
			c1 = self.cost(X, y, label)
			self.theta_[label] = self.theta_[label] - self.learning_rate * self.gradient(X, y, label)
			if abs(self.cost(X, y, label) - c1) <= self.threshold:
				break

	def train_multiclass(self, X, y):
		for label in self.unique_labels:
			self.train(X, np.array([1 if label == entry else 0 for entry in y]), label)

	def fit(self, X, y):
		self.row, self.features = X.shape
		self.unique_labels = np.unique(y)
		self.theta_ = np.zeros((len(self.unique_labels), self.features))
		self.should_normalize = self.should_normalize(X)
		if self.should_normalize:
			self.scaler_ = StandardScaler()
			X = self.scaler_.fit_transform(X)

		self.train_multiclass(X, y)

	def predict(self, X):
		if self.should_normalize:
			X = self.scaler_.transform(X)
		return np.argmax(self.sigmoid(X.dot(self.theta_.T)), axis = 1)
