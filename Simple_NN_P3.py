'''
A feed-forward NN with back propogation used to identify hand-written digits
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder

from scipy.io import loadmat

# path = os.getcwd() + 'data/ex3data1.mat'
data = loadmat('data/ex3data1.mat')

X = data['X'] # 5000 examples, 400 features
y = data['y']

print (X.shape)

# Performing one-hot encoding
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y) # Shape of (5000, 10), 5000 examples, 10 possible results, each a vector of 10 with one 'hot' value
# print (y[4999], y_onehot[4999,:])

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Theta 1 and Theta 2 are out parameter vectors for each layer

def forward_propogation(X, theta1, theta2):
    m = X.shape[0]  # No. of training examples

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # Creates array for X values with 1s in 0th col
    z2 = a1 * theta1.T

    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T

    h = sigmoid(z3) # Hypothesis

    return a1, z2, a2, z3, h

def cost (params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # Reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # Running the feed-forward pass
    a1, z2, a2, z3, h = forward_propogation(X, theta1, theta2)

    # Compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :])) # Fist term of cost function (if y = 1)
        second_term = np.multiply((1 - y[i,:]), np.log(h[i, :])) # Second term of cost function (if y = 0)

        j += np.sum(first_term - second_term) # Refer to cost full cost function (This is without regularization)

    J = J/m # Cost

    return J

# initial setup (Testing our data)
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
