import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#  Specify data path
path = '../data/ex1data1.txt'
data = pd.read_csv(path, header = None, names = ['Population', 'Profits'])

# Visualize the data
data.plot(x = 'Population', y = 'Profits', kind = 'scatter')
plt.show()

# Insert bias column and
# Split data into x (features) and y (labels)
data.insert(0, 'Ones', 1)
cols = data.shape[1]

X = data.iloc[:, 0: cols - 1]
y = data.iloc[:, cols - 1: cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([[0,0]])

# Define Cost Function
def cost(X, y, theta):
	sq_error = np.power(((X * theta.T) - y), 2)
	cost = np.sum(sq_error) / (2 * len(X))
	return cost

# Define the Gradient Descent optimizer
def grad_desc(X, y, theta, alpha=0.01, iters=1000):
	temp = np.matrix(np.zeros(theta.shape))
	parameters = int(theta.shape[1])
	cost = np.zeros(iters)

	for i in range(iters):
		error = ((X * theta.T) - y)
		for j in range(parameters):
			inner = np.multiply(error, X[:, j])
			temp[0, j] = theta[0, j] - ((alpha/len(X)) * np.sum(inner))

		theta = temp
		# cost[i] = cost(X, y, theta)
	return theta #, cost

# Test GD Function
print("Error Before: ", cost(X, y, theta))
params = grad_desc(X, y, theta)
print("Error Afterwards: ", cost(X, y, params))