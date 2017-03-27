import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import style
from sklearn.utils import shuffle

# TODO: Fix testing

style.use('ggplot')

path = os.getcwd() + '/data/mine/auto-mpg.data_2.xlsx'
df = pd.read_excel(path, 'auto-mpg', header=None,
                   names=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                          'mpg'], parse_cols=None)

iters = 5000
alpha = 0.01

# Feature normalization
df = (df - df.mean()) / df.std()

df.insert(0, 'ones', 1)
df = shuffle(df)  # Shuffles the dataframe


def cost_function(X, y, theta):
    sq_error = np.power(((X * theta.T) - y), 2)
    return np.sum(sq_error) / (2 * len(X))


def gradient_descent(X, y, theta, iters, alpha):
    temp = np.matrix(np.zeros(theta.shape))  # Creating temp variable to store each theta
    parameters = int(theta.ravel().shape[1])  # Get no of columns in theta
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))  # Calculating new values of theta

        theta = temp
        cost[i] = cost_function(X, y, theta)

    return theta, cost


rows = df.shape[0]
cols = df.shape[1]  # Gets no of columns
print cols
X = np.matrix(df.iloc[0:rows - 50, 0:cols - 1])
y = np.matrix(df.iloc[0:rows - 50, cols - 1:cols])  # Gets all rows from 4th (last) col

theta = np.matrix(np.zeros((cols - 1), dtype=np.int))

print 'Theta before: ', theta
print 'Cost before: ', cost_function(X, y, theta)

theta, cost = gradient_descent(X, y, theta, iters, alpha)
print 'Theta after: ', theta
print 'Cost after: ', cost_function(X, y, theta)


def plot(iters, cost):
    plt.plot(np.arange(iters), cost, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Error vs. Training Epoch')
    plt.show()


plot(iters, cost)

# Testing the classifier
# Test set
X2 = np.matrix(df.iloc[rows - 50:, 0:cols - 1])
y2 = np.matrix(df.iloc[rows - 50:, cols - 1:cols])

test_perf = cost_function(X2, y2, theta)

print 'Test Set Performance Cost: ', test_perf
