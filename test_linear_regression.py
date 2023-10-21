import pandas as pd
import numpy as np
from LMS.lms import stochastic_gradient_descent, batch_gradient_descent

test_directory = './../HW2/concrete-2/test.csv'
train_directory = './../HW2/concrete-2/train.csv'

# prep the data
test_data = pd.read_csv(test_directory, header=None)
train_data = pd.read_csv(train_directory, header=None)
test_data = test_data.to_numpy()
train_data = train_data.to_numpy()
ones = np.ones((len(test_data), 1))
test_data = np.column_stack((ones, test_data))
ones = np.ones((len(train_data), 1))
train_data = np.column_stack((ones, train_data))

batch_gradient_descent(r=.001, train_data=train_data, test_data=test_data)
stochastic_gradient_descent(r=.00001, train_data=train_data, test_data=test_data)

# calculate analytically
A = train_data[:,:-1]
b = train_data[:,-1]
x = np.linalg.inv(A.T@A)@A.T@b
print(x)
print(sum([(ex[-1] - np.dot(x, ex[:-1]))**2 for ex in test_data])/2)