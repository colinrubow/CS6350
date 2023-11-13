from Perceptron.perceptron import *
import numpy as np
import pandas as pd

# prep data
test_directory = './Perceptron/bank-note/test.csv'
train_directory = './Perceptron/bank-note/train.csv'
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

# get models
print('standard')
standard_perceptron, standard_errors = perceptron_standard(train_data, 10, 1e-3, test_data)
print(standard_perceptron)
print(standard_errors)

print('voted')
voted_perceptron, voted_errors = perceptron_voted(train_data, 10, 1e-3, test_data)
print(voted_perceptron)
print(voted_errors)

print('averaged')
averaged_perceptron, averaged_errors = perceptron_averaged(train_data, 10, 1e-3, test_data)
print(averaged_perceptron)
print(averaged_errors)