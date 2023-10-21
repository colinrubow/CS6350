from EnsembleLearning.bagging import Bagging
from EnsembleLearning.random_forest import RandomForest
from LMS.lms import batch_gradient_descent, stochastic_gradient_descent
import numpy as np
import pandas as pd
bagger = Bagging()
rand_forest = RandomForest()

# test_directory = './../HW2/bank-7/test.csv'
# train_directory = './../HW2/bank-7/train.csv'

attributes = {
    'age': ['numeric'],
    'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
    'marital': ['married', 'divorced', 'single'],
    'education': ['unknown', 'secondary', 'primary', 'tertiary'],
    'default': ['yes', 'no'],
    'balance': ['numeric'],
    'housing': ['yes', 'no'],
    'loan': ['yes', 'no'],
    'contact': ['unknown', 'telephone', 'cellular'],
    'day': ['numeric'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'duration': ['numeric'],
    'campaign': ['numeric'],
    'pdays': ['numeric'],
    'previous': ['numeric'],
    'poutcome': ['unknown', 'other', 'failure', 'success']
}
outputs = ('yes', 'no')

# part b
# bagger.bag(500, train_directory, test_directory, attributes, outputs, True, True, 5000)

# part c
# models = [bagger.bag(500, train_directory, test_directory, attributes, outputs, True, False, 1000) for _ in range(100)]
# first_trees = [m[0] for m in models]
# test_data = bagger.read_data(test_directory)
# # first tree predictions
# predictions_truths = [([1 if outputs[0] == f_t.predict_example(e, None, True) else -1 for f_t in first_trees], 1 if outputs[0] == e[-1] else -1) for e in test_data]
# biases = [(np.mean(p) - gt)**2 for p, gt in predictions_truths]
# vars = [np.var(p) for p, _ in predictions_truths]
# bias = np.mean(biases)
# var = np.mean(vars)
# print('single tree bias and variance:')
# print('bias: ', bias)
# print('variance: ', var)

# # bagged predictions
# predictions_truths = [([1 if outputs[0] == bagger.get_error(m, [e], outputs, True, True) else -1 for m in models], 1 if outputs[0] == e[-1] else -1) for e in test_data]
# biases = [(np.mean(p) - gt)**2 for p, gt in predictions_truths]
# vars = [np.var(p) for p, _ in predictions_truths]
# bias = np.mean(biases)
# var = np.mean(vars)
# print()
# print('bagged trees bias and variance:')
# print('bias: ', bias)
# print('variance: ', var)

# part d
# rand_forest.bag(500, train_directory, test_directory, attributes, outputs, True, True, 5000, 2, './random_forest_error_2.csv')
# rand_forest.bag(500, train_directory, test_directory, attributes, outputs, True, True, 5000, 4, './random_forest_error_4.csv')
# rand_forest.bag(500, train_directory, test_directory, attributes, outputs, True, True, 5000, 6, './random_forest_error_6.csv')

# pard e
# models = [rand_forest.bag(500, train_directory, test_directory, attributes, outputs, True, False, 1000, 2) for _ in range(100)]
# first_trees = [m[0] for m in models]
# test_data = rand_forest.read_data(test_directory)
# # first tree predictions
# predictions_truths = [([1 if outputs[0] == f_t.predict_example(e, None, True) else -1 for f_t in first_trees], 1 if outputs[0] == e[-1] else -1) for e in test_data]
# biases = [(np.mean(p) - gt)**2 for p, gt in predictions_truths]
# vars = [np.var(p) for p, _ in predictions_truths]
# bias = np.mean(biases)
# var = np.mean(vars)
# print('single tree bias and variance:')
# print('bias: ', bias)
# print('variance: ', var)

# # bagged predictions
# predictions_truths = [([1 if outputs[0] == bagger.get_error(m, [e], outputs, True, True) else -1 for m in models], 1 if outputs[0] == e[-1] else -1) for e in test_data]
# biases = [(np.mean(p) - gt)**2 for p, gt in predictions_truths]
# vars = [np.var(p) for p, _ in predictions_truths]
# bias = np.mean(biases)
# var = np.mean(vars)
# print()
# print('bagged trees bias and variance:')
# print('bias: ', bias)
# print('variance: ', var)

# problem 4
