import numpy as np
import time

from learner.Learner import Learner

class SVRG(Learner):

	def __init__(self, batch_size, gradient_computation_threshold, learning_rate, regularization, init_state):
		self.batch_size_ = batch_size
		self.gradient_computation_threshold_ = gradient_computation_threshold
		self.learning_rate_ = learning_rate
		self.regularization_ = regularization
		self.init_state_ = init_state

	def objective_function_value(self):
		c1 = 0 #0.5 * self.regularization_ * self.theta.dot(self.theta)
		c2 = 1.0 - self.y * (self.X.dot(self.theta))
		return c1 + np.sum(c2[np.where(c2 >0)[0]]) / self.rows

	def gradient_1D(self, idx, theta):
		return -1.0 * self.y[idx] * self.X[idx,:] if self.y[idx] * (self.X[idx,:].dot(theta)) < 1.0 else 0

	def gradient(self, theta):
		grad = np.zeros(self.features)
		for i in range(self.rows):
			grad = grad + self.gradient_1D(i, theta)
		return grad

	def fit(self, X, y):

		start_time = time.time()
		self.rows, self.features = X.shape
		self.X, self.y = X, y
		self.theta = np.zeros(self.features)
		current_gradient_computations = 0
		self.cost_ = []
		self.num_gradient_ = []
		num_iter = 0

		while current_gradient_computations < self.gradient_computation_threshold_:

			avg_gradient = self.gradient(self.theta) / self.rows
			current_gradient_computations = current_gradient_computations + self.rows
			cur_theta = self.theta
			num_iter = num_iter + 1

			for i in range(self.batch_size_):
				np.random.seed(self.init_state_ * (i+1) * num_iter)
				idx = np.random.randint(0, self.rows-1)
				example_avg_gradient = self.gradient_1D(idx, self.theta)
				example_cur_gradient = self.gradient_1D(idx, cur_theta)
				cur_theta = cur_theta - self.learning_rate_ * (example_cur_gradient - example_avg_gradient + avg_gradient)
				current_gradient_computations = current_gradient_computations + 1

			self.theta = cur_theta #Option 1
			self.cost_.append(self.objective_function_value())
			self.num_gradient_.append(current_gradient_computations)
			if num_iter > 1000:
				break
		self.run_time_ = int(time.time() - start_time)
		self.cost_ = np.array(self.cost_)
		self.num_gradient_ = np.array(self.num_gradient_)

	def predict(self, X, y):
		pass