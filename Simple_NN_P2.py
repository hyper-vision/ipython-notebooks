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

# Performing one-hot encoding
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y) # Shape of (5000, 10), 5000 examples, 10 possible results, each a vector of 10 with one 'hot' value
# print y[4999], y_onehot[4999,:]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Theta 1 and Theta 2 are out parameter vectors for each layer

def forward_propogation(X, theta1, theta2):
    m = X.shape[0]  # No. of training examples

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # Creates array for X values with 1s in 0th col; shape = [1, 401]
    z2 = a1 * theta1.T  # z2 shape = [1, 25]

    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)  # shape = [1, 26]
    z3 = a2 * theta2.T  # shape = [1, 10]

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

        J += np.sum(first_term - second_term) # Refer to full cost function (This is without regularization)

    J = J/m # Cost (Total average cost)
    # Regularization
    # Formula = lambda/2m * (sum of all thetas added up and squared (excluding the first column as it belongs to the bias node))
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))) 
    return J

# The derivative of the sigmoid function, aka g prime or g'
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # This section is identical to the cost function logic we already saw
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # Reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # Run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    J = 0
    delta1 = np.zeros(theta1.shape) # (25, 401)
    delta2 = np.zeros(theta2.shape) # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####
    
    
    # Performing backpropogation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        d3 = ht - yt # (1, 10) Each hypothesis - each label (Delta 3)
        
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26) (Delta 2)
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t  # Big delta1 = existing delta values + new delta values
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1/m
    delta2 = delta2/m
    
    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


# initial setup (Testing our data)
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

print 'Theta 1 shape ', theta1.shape
print 'Theta 2 shape ', theta2.shape

X = np.matrix(X)
y = np.matrix(y)
m = X.shape[0]

a1, z2, a2, z3, h = forward_propogation(X, theta1, theta2)

cost = cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

print 'Cost: ', cost
