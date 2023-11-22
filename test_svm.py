from SVM.svm import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# prep data
test_directory = './SVM/bank-note/test.csv'
train_directory = './SVM/bank-note/train.csv'
test_data = pd.read_csv(test_directory, header=None)
train_data = pd.read_csv(train_directory, header=None)
test_data = test_data.to_numpy()
for t in test_data:
    if t[-1] == 0:
        t[-1] = -1
train_data = train_data.to_numpy()
for t in train_data:
    if t[-1] == 0:
        t[-1] = -1
ones = np.ones((len(test_data), 1))
test_data = np.column_stack((ones, test_data))
ones = np.ones((len(train_data), 1))
train_data = np.column_stack((ones, train_data))

C = [100/873, 500/873, 700/873]

# problem 2
# problem 2a
learning_rate_func = lambda gamma, t: gamma/(1 + gamma*t)
for c in C:
    w, test_error, train_error = svm_primal_stochastic_descent(train_data, 100, learning_rate_func, 0.5, c, test_data)
    print('w ', w)
    print('test_error ', test_error)
    print('train_error ', train_error)

# problem 2b
learning_rate_func = lambda gamma, t: gamma/(1 + t)
for c in C:
    w, test_error, train_error = svm_primal_stochastic_descent(train_data, 100, learning_rate_func, 0.5, c, test_data)
    print('w ', w)
    print('test_error ', test_error)
    print('train_error ', train_error)

# problem 3a
def func(w, c=None, x=None):
    return np.dot(w[1:], w[1:])/2 + c*sum([np.max([0, 1 - ex[-1]*np.dot(w, ex[:-1])]) for ex in x])
w0 = np.zeros(len(train_data[0][:-1]))
for c in C:
    result = minimize(func, w0, (c, train_data))
    print(result.x)