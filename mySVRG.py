import matplotlib.pyplot as plt
import numpy as np
import sys

from preprocessor.Util import read_csv_first_col_label
from learner.SVRG import SVRG

file_path = sys.argv[1]
batch_size = int(sys.argv[2])
num_runs = int(sys.argv[3])

learning_rate = 1e-6
regularization = 1
X, y = read_csv_first_col_label(file_path)
y = np.array(map(lambda x: 1 if x == 1 else -1, y))
run_times = np.zeros(num_runs)
cost_function = []
num_gradient = []

for i in range(num_runs):
	print "Running iteration %d " % (i+1)
	svrg = SVRG(batch_size, 100 * X.shape[0], learning_rate, regularization, i+1)
	svrg.fit(X, y)
	run_times[i] = svrg.run_time_
	cost_function.append(svrg.cost_)
	num_gradient.append(svrg.num_gradient_)

"""
colors = ['g', 'r', 'b', 'c', 'm']
plt.figure()
plt.title("Change in Cost function with number of iterations")
plt.xlabel("Gradient Computations")
plt.ylabel("Cost function")
plt.grid()
for i in range(num_runs):
	plt.plot(num_gradient[i], cost_function[i], '.-', color=colors[i])
plt.savefig('tmp/SVRG_' + str(batch_size) + '.png')
"""

print "Mean Run time is %f with Standard deviation is %f " % (np.mean(run_times), np.std(run_times))