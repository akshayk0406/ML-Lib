import matplotlib.pyplot as plt
import numpy as np
import sys

from preprocessor.Util import read_csv_first_col_label
from learner.Pegasos import Pegasos

file_path = sys.argv[1]
batch_size = int(sys.argv[2])
num_runs = int(sys.argv[3])

X, y = read_csv_first_col_label(file_path)
y = map(lambda x: 1 if x == 1 else -1, y)
run_times = np.zeros(num_runs)
cost_function = []

for i in range(num_runs):
	print "Running iteration %d " % (i+1)
	pegasos = Pegasos(batch_size, 100 * X.shape[0], 1.0, i+1)
	pegasos.fit(X, y)
	run_times[i] = pegasos.run_time_
	cost_function.append(pegasos.cost_)

"""
colors = ['g', 'r', 'b', 'c', 'm']
plt.figure()
plt.title("Change in Cost function with number of iterations")
plt.xlabel("Iteration")
plt.ylabel("Cost function")
plt.grid()
for i in range(num_runs):
	plt.plot(range(len(cost_function[i])), cost_function[i], '.-', color=colors[i])
plt.savefig('tmp/Pegasos_' + str(batch_size) +'.png')
"""

print "Mean Run time is %f with Standard deviation is %f " % (np.mean(run_times), np.std(run_times))