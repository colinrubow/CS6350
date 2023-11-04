import numpy as np

def perceptron_standard(examples: list, max_num_epochs: int, learning_rate: float, test_data: list = None) -> tuple:
    # expects examples and test_data to have the format np.array([1, x, y]) for each example.   
    w = np.zeros(len(examples[0][:-1]))
    test_errors = []
    for _ in range(max_num_epochs):
        # shuffle the data
        np.random.shuffle(examples)
        for ex in examples:
            if ex[-1]*np.dot(ex[:-1], w) <= 0:
                w += learning_rate*ex[-1]*ex[:-1]
        if test_data is not None:
            test_errors.append(test_perceptron_standard(test_data, w))
            
    return w, test_errors

def perceptron_voted(examples: list, num_epochs: int, learning_rate: float, test_data: list = None) -> tuple:
    # expects examples and test_data to have the format np.array([1, x, y]) for each example.
    w = [np.zeros(len(examples[0][:-1]))]
    m = 0
    test_errors = []
    C = [1]
    for _ in range(num_epochs):
        for ex in examples:
            if ex[-1]*np.dot(w[m], ex[:-1]) <= 0:
                w[m] = w[m] + learning_rate*ex[-1]*ex[:-1]
                C[m] = 1
            else:
                C[m] += 1
        w.append(w[m])
        C.append(1)
        m += 1
        if test_data is not None:
            perceptron = [(w[i], C[i]) for i in range(len(w))]
            test_errors.append(test_perceptron_voted(test_data, perceptron))
    perceptron = [(w[i], C[i]) for i in range(len(w))]
    return perceptron, test_errors

def perceptron_averaged(examples: list, num_epochs: int, learning_rate: float, test_data: list = None) -> tuple:
    # expects examples and test_data to have the format np.array([1, x, y]) for each example.
    w = np.zeros(len(examples[0][:-1]))
    a = np.zeros(len(examples[0][:-1]))
    test_errors = []
    for _ in range(num_epochs):
        for ex in examples:
            if ex[-1]*np.dot(w, ex[:-1]) <= 0:
                w += learning_rate*ex[-1]*ex[:-1]
            a += w
        if test_data is not None:
            test_errors.append(test_perceptron_standard(test_data, a))
    return a, test_errors

def test_perceptron_standard(examples: list, perceptron: list) -> float:
    # expects examples to have the format np.array([1, x, y])
    # get num incorrect
    num_incorrect = [-1 for ex in examples if np.sign(np.dot(perceptron, ex[:-1])) != ex[-1]]
    # get num incorrect
    return len(num_incorrect)/len(examples)

def test_perceptron_voted(examples: list, perceptron: list) -> float:
    # expects examples to have the format np.array([1, x, y])
    # get predictions
    num_incorrect = [-1 for ex in examples if np.sign(np.sum([c*np.sign(np.dot(w, ex[:-1])) for w, c in perceptron])) != ex[-1]]
    # get num incorrect
    return len(num_incorrect)/len(examples)