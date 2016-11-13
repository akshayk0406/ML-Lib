import math
import numpy as np
import time

from learner.Learner import Learner

class Pegasos(Learner):

	def __init__(self, batch_size, iterations, regularization, init_state):
		self.iterations_ = iterations
		self.batch_size_ = batch_size
		self.regularization_ = regularization
		self.init_state_ = init_state

	def objective_function_value(self):
		c1, c2  = 0.5 * self.regularization_ * self.theta.dot(self.theta), 1.0 - self.y * (self.X.dot(self.theta))
		print np.sum(c2[np.where(c2 >0)[0]])
		return c1 + np.sum(c2[np.where(c2 >0)[0]]) / self.rows

	def gradient(self, selected_index):
		grad = np.zeros(self.features)
		for i, idx in enumerate(selected_index):
			grad = grad + self.y[idx] * self.X[idx,:]
		return grad

	def get_batch_for_update(self, num_iteration):
		np.random.seed(self.init_state_ * num_iteration)
		result = []
		for label, frequency in self.class_freq:
			required_entries = min(frequency, self.batch_size_)
			required_idx = np.random.choice(np.where(self.y == label)[0], required_entries, replace = False)
			for index in required_idx:
				result.append(index)
		return np.array(result)

	def fit(self, X, y):

		start_time = time.time()
		self.rows, self.features = X.shape
		self.X, self.y = X, y
		self.class_freq = zip(*np.unique(self.y, return_counts = True))
		self.theta = np.zeros(self.features)
		self.cost_ = []
		gradient_computed = 0
		num_iteration = 1

		while self.iterations_ > gradient_computed:
			
			idx = self.get_batch_for_update(num_iteration)
			required_idx = filter(lambda x: self.y[x] * (self.X[x,:].dot(self.theta)) < 1.0, idx)
			learning_rate = 1.0 / (self.regularization_ * num_iteration)
			self.theta = (1.0 - 1.0 / num_iteration) * self.theta + (learning_rate * self.gradient(required_idx) * 1.0) / self.batch_size_
			gradient_computed = gradient_computed + len(required_idx)
			self.theta = min(1.0, (1.0/(np.sqrt(self.regularization_) * np.linalg.norm(self.theta)))) * self.theta
			#print self.objective_function_value()
			self.cost_.append(self.objective_function_value())
			num_iteration = num_iteration + 1
			if num_iteration >= 1001:
				break

		self.run_time_ = int(time.time() - start_time)
		self.cost_ = np.array(self.cost_)

	def predict(self, X):
		pass