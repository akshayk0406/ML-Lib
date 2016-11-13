from abc import ABCMeta, abstractmethod

class PreProcessor(object):

	__metaclass__ = ABCMeta
	@abstractmethod
	def fit(self, X):
		pass

	@abstractmethod
	def transform(self, X):
		pass

	@abstractmethod
	def fit_transform(self, X):
		pass
