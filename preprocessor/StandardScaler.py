import numpy as np

from preprocessor.PreProcessor import PreProcessor

class StandardScaler(PreProcessor):

	def __init__(self):
		self.mean_ = []
		self.var_ = []

	def fit(self, X):
		self.mean_ = np.mean(np.array(X), axis = 0)
		self.var_ = np.var(np.array(X), axis = 0)

	def transform(self, X):
		
		rows = len(X)
		cols = len(X[0])

		for i in range(0, rows):
			for j in range(0, cols):
				X[i][j] = ((X[i][j] - self.mean_[j]) * 1.0) / self.var_[j]

		return X

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)