import numpy as np
from typing import Callable

def svm_primal_stochastic_descent(examples: list, max_num_epochs: int, learning_rate_func: Callable, initial_learning_rate: float, hyperparameter: float, test_data: list) -> tuple:
    # expects examples and test_data to have the format np.array([1, x, y]) for each example.
    w = np.zeros(len(examples[0][:-1]))
    for t in range(max_num_epochs):
        # shuffle it bro!
        np.random.shuffle(examples)
        for ex in examples:
            if ex[-1]*np.dot(w, ex[:-1]) <= 1:
                w = w - initial_learning_rate*(np.array([0] + w[1:].tolist()) - hyperparameter*len(examples)*ex[-1]*ex[:-1])
            else:
                w = (1 - initial_learning_rate)*np.array([w[0]/(1-initial_learning_rate)] + w[1:].tolist())
        # new learning rate
        initial_learning_rate = learning_rate_func(initial_learning_rate, t)
    test_error = test_svm(test_data, w)
    train_error = test_svm(examples, w)
    return w, test_error, train_error

def test_svm(examples: list, svm: list) -> float:
    # expects examples to have the format np.array([1, x, y])
    # get num incorrect
    num_incorrect = [-1 for ex in examples if np.sign(np.dot(svm, ex[:-1])) != ex[-1]]
    # get num incorrect
    return len(num_incorrect)/len(examples)
