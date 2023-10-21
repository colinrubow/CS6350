import numpy as np
import csv

def batch_gradient_descent(epsilon: float = 10e-6, r: float = 0.1, train_data: list = None, test_data: list = None):
    """
    Runs gradient descent with a batch style.

    Parameters
    ----------
    epsilon : the stopping criteria. If the norm of the change of the trained model is below epsilon we have 'converged'
    r : the learning rate. 0.001 seems to work ok
    train_data : the training examples. The first column needs to be all 1's. The last column is the characterization of the example.
    test_data : the testing examples. The first column needs to be all 1's. The last column is the characterization of the example.

    Outputs
    -------
    prints : the trained model. The test error of the trained model.
    writes : a csv file recording the testing and training errors for each iteration.
    """
    # w[0] is the constant term
    w_old = np.array([0.0]*(len(train_data[0]) - 1))
    w_new = np.array([1.0]*(len(train_data[0]) - 1))
    error = sum([(ex[-1] - np.dot(w_old, ex[:-1]))**2 for ex in train_data])/2
    costs_train = [error]
    costs_test = [sum([(ex[-1] - np.dot(w_old, ex[:-1]))**2 for ex in test_data])/2]
    while np.linalg.norm(w_old - w_new) >= epsilon:
        w_old = w_new
        delta_J = np.array([-sum([(ex[-1] - np.dot(w_old, ex[:-1]))*e[i] for i, ex in enumerate(train_data)]) for e in np.array(train_data).T[:-1]])
        w_new = w_old - r*delta_J
        error = sum([(ex[-1] - np.dot(w_new, ex[:-1]))**2 for ex in train_data])/2
        costs_train.append(error)
        costs_test.append(sum([(ex[-1] - np.dot(w_new, ex[:-1]))**2 for ex in test_data])/2)
    print(w_new)
    print(error)
    header = ['cost_train', 'cost_test']
    data = [[costs_train[i], costs_test[i]] for i in range(len(costs_train))]
    with open('./batch_gradient_descent.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
def stochastic_gradient_descent(epsilon: float = 10e-6, r: float = 0.1, train_data: list = None, test_data: list = None):
    """
    Runs gradient descent with a stochastic style.

    Parameters
    ----------
    epsilon : the stopping criteria. If the norm of the change of the trained model is below epsilon we have 'converged'
    r : the learning rate. 0.00001 seems to work ok
    train_data : the training examples. The first column needs to be all 1's. The last column is the characterization of the example.
    test_data : the testing examples. The first column needs to be all 1's. The last column is the characterization of the example.

    Outputs
    -------
    prints : the trained model. The test error of the trained model.
    writes : a csv file recording the testing and training errors for each iteration.
    """
    # w[0] is the constant term
    w_old = np.array([0.0]*(len(train_data[0]) - 1))
    w_new = np.array([1]*(len(train_data[0]) - 1))
    error = sum([(ex[-1] - np.dot(w_old, ex[:-1]))**2 for ex in train_data])/2
    costs_train = [error]
    costs_test = [sum([(ex[-1] - np.dot(w_old, ex[:-1]))**2 for ex in test_data])/2]
    while np.linalg.norm(w_old - w_new) >= epsilon:
        for x in train_data:
            w_old = w_new
            grad = x[-1] - np.dot(w_old, x[:-1])
            w_new = np.array([weight + r*grad*x[j] for j, weight in enumerate(w_old)])
        error = sum([(ex[-1] - np.dot(w_new, ex[:-1]))**2 for ex in train_data])/2
        costs_train.append(error)
        costs_test.append(sum([(ex[-1] - np.dot(w_new, ex[:-1]))**2 for ex in test_data])/2)
    print(w_new)
    print(error)
    header = ['cost_train', 'cost_test']
    data = [[costs_train[i], costs_test[i]] for i in range(len(costs_train))]
    with open('./stochasitc_gradient_descent.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)