# 0. Initialize learning rate and iterations
# 1. Get the data
# 2. Manipulate the data
# 3. Create data variables
# 4. Define cost function
# 5. Optimized the data (Gradient Descent)
# 6. Creating the plotting data
# 7. Plotting the data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import style

style.use('ggplot')

alpha = 0.01
iters = 1000

file_path = os.getcwd() + '\data\ex1data1.txt'
df = pd.read_csv(file_path, header = None, names = ['Population', 'Profits'])

df.insert(0, 'ones', 1)

X = np.matrix(df[['ones', 'Population']])
y = np.matrix(df[['Profits']])
theta = np.matrix([[0,0]])
m = len(X)

def cost_func(X, y, theta):
    sq_error = np.power(((X * theta.T) - y), 2)
    return np.sum(sq_error)/(2 * m)

print "Cost = ", cost_func(X, y, theta), 'when theta = ', theta

def gradient_descent(X, y, iters, alpha, theta, m):
    temp = np.matrix(np.zeros(theta.shape)) # Creating temp variable to store each theta
    theta_count = len(theta.ravel().T) # sets theta_count = 2 (No. of cols in theta)
    cost = np.zeros(iters) # Creates a zero vector to store costs for each iteration
    
    for i in range(iters): # Runs gradient descent for n number of iterations
        error = (X * theta.T) - y # Returns vector of errors

        for j in range(theta_count): # Iterates through each column
            each_hypo = np.multiply(error, X[:, j]) # Multiplying error vector * all rows of X in j'th column
                                                    # (1st then 2nd)
            temp[0, j] = theta[0, j] - ((alpha / m) * np.sum(each_hypo))
            # print 'temp', temp
            
        theta = temp
        # print '\ntheta', theta
        cost[i] = cost_func(X, y, theta)
        
    return theta, cost

theta, cost = gradient_descent(X, y, iters, alpha, theta, m)

x = np.linspace(df.Population.min(), df.Population.max(), 100) # Creates array of 100 evenly spaced numbers
                                                               # between minimum and max val of Population
hypothesis = theta[0,0] + theta[0,1] * x

def plot_data(x, hypothesis, iters):
    # Plotting the Data and Regression Line
    plt.scatter(df['Population'], df['Profits'])
    plt.plot(x, hypothesis)
    plt.xlabel('Population (Millions)')
    plt.ylabel('Profit (K)')
    plt.legend(loc=4)
    plt.show()
    
    # Plotting the cost function
    plt.plot(np.arange(iters), cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost J(theta)')
    plt.show()
    
plot_data(x, hypothesis, iters)

print "Cost = ", cost_func(X, y, theta), 'when theta = ', theta