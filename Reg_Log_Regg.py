import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

path = os.getcwd() + '\data\ex2data2.txt'
df = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

def plot(df):
    pos = df[df['Accepted'].isin([1])] # Taking all the rows where y = 1
    neg = df[df['Accepted'].isin([0])] # "      "   "   "    "     y = 0
    
    plt.scatter(pos['Test 1'], pos['Test 2'], s = 20, c='b', label='Accepted')
    plt.scatter(neg['Test 1'], neg['Test 2'], s = 30, c='r', marker='x', label='Rejected')
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    plt.legend()
    plt.show()
    
