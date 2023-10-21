from DecisionTree.ID3 import ID3
import numpy as np
import csv
import random

class RandomForest():
    def __init__(self) -> None:
        pass

    def bag(self, num_trees: int, directory_train: str, directory_test: str, attributes: dict, outputs: tuple, contains_numeric: bool = False, log: bool = False, num_samples: int = 1000, num_attributes: int = None, filename_output: str = None):
        """
        trains and writes a csv of the random_forest errors.

        num_trees : the number of trees in the forest
        directory_train : the directory of the training data
        directory_test : the directory of the tresting data
        attributes : a dictionary where the keys are attributes and values are lists of values the attributes may take
        outputs : a tuple of the classification the data takes (i.e. ('yes', 'no')). The first value is the positive classification if binary
        contains_numeric : whether an attribute is numeric or not
        log : whether to record training and testing error as the model is trained.
        num_samples : the number of samples to randomly choose from the training data each iteration
        num_attributes : the number of attributes to randomly sample from when building trees
        """
        # read in data
        train_data = self.read_data(directory_train)
        test_data = self.read_data(directory_test)
        # modify outputs to that outputs[1] is the positive classification and outputs[-1] is the negative classification
        outputs = [None, outputs[0], outputs[1]]

        # set up logging
        if log:
            train_error = []
            test_error = []
        classifiers = []
        # start iterating
        for _ in range(num_trees):
            tree = ID3()
            # get samples
            train_data_sampled = random.choices(train_data, k=num_samples)
            train_data_sampled = [s.copy() for s in train_data_sampled]
            # train the tree
            tree.train_model(None, 'IG', train_data_sampled, attributes, contains_numeric, num_attributes=num_attributes)
            # store the tree
            classifiers.append(tree)
            # loggit
            if log:
                train_error.append(self.get_error(classifiers, train_data, outputs, contains_numeric, False))
                test_error.append(self.get_error(classifiers, test_data, outputs, contains_numeric, False))
        # save data in csv
        if log:
            # prep data
            header = ['train_error', 'test_error']
            error_data = [[train_error[i], test_error[i]] for i in range(len(train_error))]
            with open(filename_output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(error_data)
        return classifiers

    def get_error(self, models: list, data: list, outputs: tuple, contains_numeric: bool, get_prediction: bool = False):
        num_incorrect = 0
        for example in data:
            guess = 0
            for tree in models:
                sub_guess = 1 if outputs[1] == tree.predict_example(example, contains_numeric=contains_numeric) else -1
                guess += sub_guess
            guess = int(np.sign(guess))
            if guess == 0:
                guess = 1
            if get_prediction: return guess
            if example[-1] != outputs[guess]:
                num_incorrect += 1
        return num_incorrect/len(data)

    @staticmethod
    def read_data(directory: str) -> list:
        """
        Loading the training/testing data.

        Arguments
        ---------
        directory : the directory of the training/testing data

        Returns
        -------
        The training/testing data as a 2D list.
        """
        with open(directory, 'r') as f:
           data = [line.strip().split(',') for line in f]
        return data