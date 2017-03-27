import os
import numpy as np
import scipy.optimize as opt
import pandas as pd

path = os.getcwd() + '\data\ex2data1.txt'
df = pd.read_csv(path, header = None, names = ["Exam 1", "Exam 2", "Admitted"])
df.insert(0, 'ones', 1)

cols = df.shape[1]
X = np.array(df.iloc[:,0:cols-1])
y = np.array(df.iloc[:, cols-1:cols])
theta = np.zeros(3)

def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
def cost(theta, X, y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second)/ len(X)

def gradient(theta, X, y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
        
    return grad

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y)) # Learning the Parameters
# resutl[0] are the new theta values

def predict(theta, X):
    prediction = (X * theta.T)
    return[1 if x >= 0.5 else 0 for x in prediction]
    pass

new_theta = np.matrix(result[0])
predictions = predict(new_theta, X)
# Creates a list, compares predictions to y, 1 if match, 0 otherwise
correct = [1 if ((a==1 and b==1) or (a==0 and b==0)) else 0 for (a,b) in zip(predictions, y)]
accuracy = (sum(correct) * 100)/len(correct)
# accuracy = (sum(correct) / len(correct)))
print accuracy, '%'
