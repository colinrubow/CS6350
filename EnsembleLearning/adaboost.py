from DecisionTree.ID3 import ID3
import numpy as np
import csv

class AdaBoost():
    def __init__(self) -> None:
        pass
    
    def train_model(self, num_iterations: int, directory_train: str, attributes: dict, outputs: tuple, contains_numeric: bool = False, log: bool = False, directory_test: str = None) -> None:
        # first construct D, which will be of length of the train data and each value will be 1/len(train data)
        train_data = self.__read_data(directory_train)
        test_data = self.__read_data(directory_test)
        length_train_data = len(train_data)
        length_test_data = len(test_data)
        # so that outputs[1] is the positive classification and outputs[-1] is the negative classification
        outputs = [None, outputs[0], outputs[1]]
        D = [1/length_train_data]*length_train_data

        # start iterating
        if log:
            iteration_train_error = []
            iteration_test_error = []
            overall_train_error = []
            overall_test_error = []
        classifiers = []
        classifier_votes = []
        for _ in range(num_iterations):
            model_i = ID3()
            # build a classifier using ID3
            # multiply examples to fit D distribution
            #### FIXME
            min_d = 1/min(D)
            multipliers = [d*min_d for d in D]
            # multiply the examples
            train_data_new = []
            for i, m in enumerate(multipliers):
                for _ in range(int(m)):
                    train_data_new.append(train_data[i])
            # write new train data to new csv (bleah this efficiency is great :!)
            with open('./temp.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(train_data_new)
            directory_train_new = './temp.csv'
            model_i.train_model(2, 'IG', directory_train_new, attributes, contains_numeric)
            # FIXME
            #####
            train_accuracy_i = model_i.test_model(directory_train, contains_numeric, False)
            test_accuracy_i = model_i.test_model(directory_test, contains_numeric, False)
            # compute its vote
            vote = np.log(train_accuracy_i/(1 - train_accuracy_i))/2
            # compute new D
            match_list = np.array([1 if example[-1] == model_i.predict_example(example, contains_numeric=contains_numeric) else -1 for example in train_data])
            D = np.multiply(np.exp(-vote*match_list), D)
            D = D/sum(D)
            # save classifier data
            classifiers.append(model_i)
            classifier_votes.append(vote)

            if log:
                iteration_train_error.append(1 - train_accuracy_i)
                iteration_test_error.append(1 - test_accuracy_i)
                num_incorrect = 0
                for example in train_data:
                    guess = 0
                    for i, c in enumerate(classifiers):
                        sub_guess = 1 if outputs[1] == c.predict_example(example, contains_numeric=contains_numeric) else -1
                        guess += classifier_votes[i]*sub_guess
                    guess = int(np.sign(guess))
                    if example[-1] != outputs[guess]:
                        num_incorrect += 1
                overall_train_error.append(num_incorrect/length_train_data)
                num_incorrect = 0
                for example in test_data:
                    guess = 0
                    for i, c in enumerate(classifiers):
                        sub_guess = 1 if outputs[1] == c.predict_example(example, contains_numeric=contains_numeric) else -1
                        guess += classifier_votes[i]*sub_guess
                    guess = int(np.sign(guess))
                    if example[-1] != outputs[guess]:
                        num_incorrect += 1
                overall_test_error.append(num_incorrect/length_test_data)
        # save log data if desired
        if log:
            header = ['iteration_test_error', 'iteration_train_error', 'overall_test_error', 'overall_train_error']
            data = [[iteration_test_error[i], iteration_train_error[i], overall_test_error[i], overall_train_error[i]] for i in range(len(overall_test_error))]
            with open('./adaboost_errors.csv', 'w', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(header)
                csv_writer.writerows(data)


    @staticmethod
    def __read_data(directory: str) -> list:
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
