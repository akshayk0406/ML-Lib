import math
import numpy as np
import time

from learner.Learner import Learner

class SMO(Learner):

	def __init__(self, tolerance, max_iterations, regularization, random_seed):
		self.tolerance_ = tolerance
		self.max_iterations_ = max_iterations
		self.regularization_ = regularization
		self.random_seed_ = random_seed

	def compute_similarity(self, x, y):
		return np.dot(x,y)

	def compute_cost(self):
		lagrange_mul = self.alpha_.reshape(self.rows,1)
		return np.sum(self.Q * (lagrange_mul.dot(lagrange_mul.T))) / 2.0 - np.sum(self.alpha_)

	def compute_kernel_matrix(self):
		self.Q  = np.zeros((self.rows, self.rows))
		for i in range(self.rows):
			for j in range(self.rows):
				self.Q[i][j] = self.y[i] * self.y[j] * self.compute_similarity(self.X[i], self.X[j])

	def is_up(self, x):
		b1 =  (self.alpha_[x] < self.regularization_[x] and self.y[x] == 1)
		b2 =  (self.alpha_[x] > 0 and self.y[x] == -1)
		return (b1 or b2)

	def is_low(self, x):
		b1 =  (self.alpha_[x] < self.regularization_[x] and self.y[x] == -1)
		b2 =  (self.alpha_[x] > 0 and self.y[x] == 1)
		return (b1 or b2)

	def find_i(self):
		i = 0
		max_value = -1e30
		for idx in range(self.rows):
			if self.is_up(idx):
				val = -1.0 * self.y[idx] * self.grad_[idx]
				if val > max_value:
					max_value = val
					i = idx
		return i

	def find_j(self, i):
		j = 0
		min_value = 1e30
		for idx in range(self.rows):
			if self.is_low(idx) and (self.y[idx] * self.grad_[idx] > self.y[i] * self.grad_[i]):
				prod = self.y[i] * self.y[idx]
				a = max(self.tau_, self.Q[i][i] + self.Q[idx][idx] - 2 * (self.Q[i][idx] / prod))
				b = -1.0 * ((-1.0 * self.y[i] * self.grad_[i] + self.y[idx] * self.grad_[idx]) ** 2)
				val = b / a
				if val < min_value:
					min_value = val
					j = idx
		return j

	def find_working_set(self):
		i = self.find_i()
		j = self.find_j(i)
		return i, j

	def update(self, i, j):
		prod = self.y[i] * self.y[j]
		a = max(self.tau_, self.Q[i][i] + self.Q[j][j] - 2 * (self.Q[i][j] / prod))
		if self.y[i] != self.y[j]:
			self.alpha_[i] = self.alpha_[i] - (self.grad_[i] + self.grad_[j]) / a
			self.alpha_[j] = self.alpha_[j] - (self.grad_[i] + self.grad_[j]) / a
		else:
			self.alpha_[i] = self.alpha_[i] + (self.grad_[j] - self.grad_[i]) / a
			self.alpha_[j] = self.alpha_[j] + (self.grad_[i] - self.grad_[j]) / a

	def restrict(self, i, j):
		if self.y[i] != self.y[j]:
			diff = self.alpha_[i] - self.alpha_[j]
			if diff < 0:
				if self.alpha_[j] < 0:
					self.alpha_[j], self.alpha_[i] = 0, diff
			else:
				if self.alpha_[i] < 0:
					self.alpha_[i], self.alpha_[j] = 0, -1.0 * diff

			if diff > self.regularization_[i] - self.regularization_[j]:
				if self.alpha_[i] > self.regularization_[i]:
					self.alpha_[i], self.alpha_[j] = self.regularization_[i], self.regularization_[i] - diff
			else:
				if self.alpha_[j] > self.regularization_[j]:
					self.alpha_[i], self.alpha_[j] = self.regularization_[j] + diff, self.regularization_[j]
		else:
			csum = self.alpha_[i] + self.alpha_[j]
			if csum > self.regularization_[i]:
				if self.alpha_[i] > self.regularization_[i]:
					self.alpha_[i], self.alpha_[j] = self.regularization_[i], csum - self.regularization_[i]
			if csum > self.regularization_[j]:
				if self.alpha_[j] > self.regularization_[j]:
					self.alpha_[j], self.alpha_[i] = self.regularization_[j], csum - self.regularization_[j]
			if csum < self.regularization_[i]:
				if self.alpha_[j] < 0:
					self.alpha_[j], self.alpha_[i] = 0, csum
			if csum < self.regularization_[j]:
				if self.alpha_[i] < 0:
					self.alpha_[i], self.alpha_[j] = 0, csum

	def normalize(self):
		for i in range(self.rows):
			norm = np.linalg.norm(self.X[i])
			self.X[i] = self.X[i] / norm

	def fit(self, X, y):

		np.random.seed(self.random_seed_)
		self.cost_ = []
		start_time = time.time()
		self.rows, self.features = X.shape
		self.X, self.y = X, y
		self.normalize()
		print "Computing kernel matrix .... "
		self.compute_kernel_matrix()
		self.alpha_ = np.zeros(self.rows)
		self.grad_ = self.Q.dot(self.alpha_) - np.ones(self.rows)
		self.tau_ = 1e-12
		init_cost = self.compute_cost()
		self.cost_.append(init_cost)
		print "Initial cost %f "  % (init_cost)
		num_iteration = 0

		while num_iteration < self.max_iterations_:
			i, j = self.find_working_set()
			self.update(i, j)
			self.restrict(i, j)
			self.grad_ = self.Q.dot(self.alpha_) - np.ones(self.rows)
			updated_cost = self.compute_cost()
			self.cost_.append(updated_cost)
			if init_cost - updated_cost < self.tolerance_:
				print "breaking on iteration %d " % (num_iteration)
				break
			init_cost = updated_cost
			num_iteration = num_iteration + 1
			if num_iteration%50 == 0:
				print "Iteration %d with Cost %f " % (num_iteration, updated_cost)

		self.run_time_ = int(time.time() - start_time)

	def predict(self, X):
		pass