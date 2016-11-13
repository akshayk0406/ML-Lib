class Node(object):

	def __init__(self, attr, threshold, isLeaf = False, label = -1, left = None, right = None):

		self.attr = attr
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.label = label
		self.left = left
		self.right = right