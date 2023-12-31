import json
from collections import Counter
import numpy as np
from scipy.stats import entropy
from typing import Callable
import copy
import random as r

class ID3():
    """
    A class for running the ID3 decision tree machine learning algorithm. To use:
    1) create the instance with my_tree = ID3()
    2) train the model with my_tree.train_model(*args)
    note that attributes is a dictionary where the keys are the different attributes and the values are the different values that attribute may take on.
    If the attribute's values are numeric, simply make that attribute's value the string 'numeric' and set the function's argument contains_numeric to True.
    If the dataset has values that are unknown there are two methods for handling this. The first is to include the string 'unknown' as a value in the attribute list.
    The second way is to not include the string 'unknown' as a value in the attribute list. Any value that appears in the dataset, that is not included in the
    attribute's list will be handled by assuming the most common value. When testing the model, the contains_unknown argument is True when the string
    'unknown' is not included in the attribute's values when the model was trained.

    Methods
    -------

    """
    def __init__(self) -> None:
        self.model = None

    def __read_data(self, directory: str) -> list:
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

    def test_model(self, directory: str, contains_numeric: bool = False, contains_unknown: bool = False) -> float:
        """
        Test the model.

        Arguments
        ---------
        directory : the directory of the testing data

        Returns
        -------
        The percentage of the accuracy of the model against the test data.
        """
        testing_data = self.__read_data(directory)
        if contains_unknown:
            # fill in the unknown data with the mode of that value
            for i in range(len(testing_data[0]) - 1):
                # iterate over the attributes
                real_values = [d[i] for d in testing_data if not d[i] == 'unknown']
                # assuming all attributes don't have all examples as unknown (why have this attribute if that's the case?)
                if len(real_values) < len(testing_data):
                    # some of the data is unknown
                    # get mode of this data
                    counts = Counter(real_values)
                    value_mode = counts.most_common(1)[0][0]
                    for d in testing_data:
                        if d[i] == 'unknown': d[i] = value_mode
                # data for this attribute is complete now
            # data is complete now
        num_tests = len(testing_data)
        match_list = [1 if example[-1] == self.predict_example(example, contains_numeric=contains_numeric) else 0 for example in testing_data]
        score = sum(match_list)
        return score/num_tests

    def predict_example(self, example: list, model: dict = None, contains_numeric: bool = False) -> str:
        """
        Runs an example through the model and returns the prediction.

        Arguments
        ---------
        example : a list of the values the attributes take.

        model : the dictionary tree to access

        Returns
        -------
        The model's prediction
        """
        if type(model) == str: return model
        if not model: return self.predict_example(example, self.model, contains_numeric)
        i = list(model.keys())[0]
        # numeric case
        if contains_numeric and list(model[i].keys())[0][0] in ['>', '<']:
            thresh = float(list(model[i].keys())[0][1:])
            if float(example[i]) >= thresh:
                return self.predict_example(example, model[i]['>' + str(thresh)], contains_numeric)
            else:
                return self.predict_example(example, model[i]['<' + str(thresh)], contains_numeric)
        return self.predict_example(example, model[i][example[int(i)]], contains_numeric)

    def load_model(self, directory: str) -> None:
        """
        Loads a model. The model is a dictionary saved in a .txt file.

        Arguments
        ---------
        directory : the directory of the saved model.
        """
        with open(directory, 'r') as f:
            dict_str = f.read()
        self.model = json.loads(dict_str)

    def save_model(self, directory: str) -> None:
        """
        Saves the model to the given directory as the string format in txt file.

        Arguments
        ---------
        directory : the given directory (.txt)
        """
        dict_str = json.dumps(self.model)
        with open(directory, 'w') as f:
            f.write(dict_str)

    def train_model(self, max_depth: int = None, gain_type: str = 'IG', training_data: list = None, attributes: dict = None, contains_numeric: bool = False, num_attributes: int = None) -> None:
        """
        Given some training data, will train a model.

        Arguments
        ---------
        max_depth : the depth the tree is allowed to go to.
        gain_type : the criteria for determining which attribute to split with.
            'IG' -> Information Gain
            'ME' -> Majority Error
            'GI' -> Gini Index
        directory : the directory of the training data.
        attributes : the attributes as keys and a list of values as values.
        """
        # training_data = self.__read_data(directory)
        match gain_type:
            case 'IG' : gain_function = self.__entropy
            case 'ME' : gain_function = self.__majority_error
            case 'GI' : gain_function = self.__gini_index
        self.__attributes_list = list(attributes.keys())
        self.model = self.__run_ID3(training_data, attributes.copy(), gain_function, 0, max_depth, contains_numeric, num_attributes)

    def __run_ID3(self, s : list, attributes : dict, gain_function: Callable, current_depth : int, depth_limit: int = None, contains_numeric: bool = False, num_attributes: int = None) -> dict:
        """
        Runs the ID3 algorithm.

        Arguments
        ---------

        Returns
        -------
        The model
        """
        s_labels = [ex[-1] for ex in s]
        # if all labels are the same
        if all([s_labels[0] == label for label in s_labels]):
            return s[0][-1]
        # if out of attributes or depth limit is reached
        if not attributes or current_depth == depth_limit:
            counts = Counter(s_labels)
            return counts.most_common(1)[0][0]
        current_depth += 1
        best_attribute = self.__get_best_attribute(copy.deepcopy(s), attributes=copy.deepcopy(attributes), gain_function=gain_function, contains_numeric=contains_numeric, num_attributes=num_attributes)
        best_attribute_index = list(attributes.keys()).index(best_attribute)
        overall_attribute_index = self.__attributes_list.index(best_attribute)
        node = {overall_attribute_index: {}}
        # if contains_numeric and values is 'numeric'
        values = attributes.pop(best_attribute)
        if contains_numeric and values[0] == "numeric":
                ex_values = [float(ex[best_attribute_index]) for ex in s]
                thresh = np.median(ex_values)
                values = ['>' + str(thresh), '<' + str(thresh)]
                for es in s:
                    es[best_attribute_index] = values[0] if float(es[best_attribute_index]) >= thresh else values[1]
        else:
            # check for example values that aren't in values list
            real_values = [ex[best_attribute_index] for ex in s if ex[best_attribute_index] in values]
            if not real_values:
                # all values in s are not in values, return node with most common outcome
                counts = Counter(s_labels)
                return counts.most_common(1)[0][0]
            if len(real_values) < len(s):
                # there are some values in s not in the list. Correct them.
                counter = Counter(real_values)
                value_mode = counter.most_common(1)[0][0]
                for ex in s:
                    if ex[best_attribute_index] not in values: ex[best_attribute_index] = value_mode
        for v in values:
            sv = [ex[:best_attribute_index] + ex[best_attribute_index+1:] for ex in s if ex[best_attribute_index] == v]
            if not sv:
                counts = Counter(s_labels)
                result = counts.most_common(1)[0][0]
            else:
                result = self.__run_ID3(copy.deepcopy(sv), copy.deepcopy(attributes), gain_function, current_depth, depth_limit, contains_numeric)
            node[overall_attribute_index].update({v: result})
        return node

    def __get_best_attribute(self, s : list, attributes : dict, gain_function : Callable, contains_numeric: bool = False, num_attributes: int = None) -> str:
        """
        Determines the best attribute to split s with.

        Arguments
        ---------
        s : the examples
        attributes : the attributes with values
        gain_type : Information gain, majority error, or gini index

        Returns
        -------
        The attribute that best splits the data
        """
        current_performance = gain_function(s)
        keys = list(attributes.keys())
        # keys is ordered so I need to make pairings and then extract with random the num of attributes to bounce wit
        keys = [(key, k) for k, key in enumerate(keys)]
        if num_attributes is not None:
            keys = r.sample(keys, k=num_attributes)
        best_attribute = keys[0][0]
        best_performance = 0
        # I'm pretty sure attributes.keys() is ordered
        for a, i in keys:
            values = attributes[a]
            if contains_numeric and values[0] == "numeric":
                ex_values = [float(ex[i]) for ex in s]
                thresh = np.median(ex_values)
                values = ['upper', 'lower']
                for es in s:
                    es[i] = 'upper' if float(es[i]) >= thresh else 'lower'
            else:
                # check for example values that aren't in values list
                real_values = [ex[i] for ex in s if ex[i] in values]
                if not real_values:
                    # all values in s are not in list. move on.
                    continue
                if len(real_values) < len(s):
                    # there are some values in s not in the list. Correct them with most often occuring value
                    counter = Counter(real_values)
                    value_mode = counter.most_common(1)[0][0]
                    for ex in s:
                        if ex[i] not in values: ex[i] = value_mode
            sets = [[] for _ in values]
            for ex in s:
                sets[values.index(ex[i])].append(ex)
            split_performance = sum([len(set)*gain_function(set) for set in sets])/len(s)
            performance_increase = current_performance - split_performance
            if performance_increase > best_performance: best_attribute = a; best_performance = performance_increase
        return best_attribute

    def __entropy(self, sv: list) -> float:
        """
        calculates the entropy of a set

        Arguments
        ---------
        sv : the set

        Returns
        -------
        the entropy
        """
        # only the last index of each list in sv is needed
        labels = [s[-1] for s in sv]
        value, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=2)

    def __majority_error(self, sv: list) -> float:
        """
        calculates the majority error of a set

        Arguments
        ---------
        sv : the set

        Returns
        -------
        the majority error
        """
        # only the last index of each list in sv is needed
        if not sv: return 0
        labels = [s[-1] for s in sv]
        counts = Counter(labels)
        majority_label, majority_count = counts.most_common(1)[0]
        return (len(sv) - majority_count)/len(sv)

    def __gini_index(self, sv: list) -> float:
        """
        calculates the gini index of a set

        Arguments
        ---------
        sv : the set

        Returns
        -------
        the gini index
        """
        # only the last index of each list in sv is needed
        labels = [s[-1] for s in sv]
        counts = Counter(labels)
        counts = counts.values()
        return 1 - sum([(c/len(sv))**2 for c in counts])