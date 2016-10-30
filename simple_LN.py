import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from matplotlib import style

style.use('ggplot')

path = os.getcwd() + '\data\ex1data1.txt'
df = pd.read_csv(path, header = None, names = ['Population','Profit'])
df.insert(0, 'ones', 1) # Adding(appending) col of 1's as first col
df_info = df.describe()

X = np.matrix(df[['ones', 'Population']])
y = np.matrix(df[['Profit']])
theta = np.matrix(np.array([0,0]))

def plot_data(x, hypo, iters):
    # Plotting the Data and Regression Line
    plt.scatter(df['Population'], df['Profit'])
    plt.plot(x, hypo)
    plt.xlabel('Population (Millions)')
    plt.ylabel('Profit (K)')
    plt.legend(loc=4)
    plt.show()
    
    # Plotting the cost function
    plt.plot(np.arange(iters), cost, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost J(theta)')
    plt.show()

# plot_data()

# Cost function J(theta) i.e (sum(hypo(X) - y)**2)/(2 * m) for all X and y, where m = len(dataset)
def cost_function(X, y, theta):
    sq_error = np.power(((X * theta.T)-y), 2)
    return np.sum(sq_error) / (2 * len(X))

# Before optimization
print 'Cost: ', cost_function(X, y, theta), ' with current value of theta as: ', theta 

def gradient_descent(X, y, theta, iters, alpha):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0,j] = theta[0, j] - ((alpha/len(X)) * np.sum(term)) # Calculating new values of theta
            
        theta = temp
        cost[i] = cost_function(X, y, theta)
        
    return theta, cost

alpha = 0.01 # Learning rate
iters = 2000 # Iterations

theta, cost = gradient_descent(X, y, theta, iters, alpha)

# After optimization
print 'Cost: ', cost_function(X, y, theta), ' with current value of theta as: ', theta

# Creating the plotting data
x = np.linspace(df.Population.min(), df.Population.max(), 100)
hypo = theta[0,0] + (theta[0,1] * x) # Hypo = theta_0 + theta_1 * x

plot_data(x, hypo, iters)