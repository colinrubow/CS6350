import json
from collections import Counter

class ID3():
    """
    A class for running the ID3 decision tree machine learning algorithm.

    Methods
    -------

    """
    def __init__(self) -> None:
        self.model = None

    def read_data(self, directory: str) -> list:
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
    
    def test_model(self, directory: str) -> float:
        """
        Test the model.

        Arguments
        ---------
        directory : the directory of the testing data

        Returns
        -------
        The percentage of the accuracy of the model against the test data.
        """
        testing_data = self.read_data(directory)
        num_tests = len(testing_data)
        match_list = [1 if example[-1] == self.predict_example(example[:-1]) else 0 for example in testing_data]
        score = sum(match_list)
        return score/num_tests

    def predict_example(self, example: list, model: dict = None) -> str:
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
        if not example: return model
        if not model: return self.predict_example(example, self.model)
        return self.predict_example(model[example[0]], example[1:])
        
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
        dict_str = str(self.model)
        with open(directory, 'w') as f:
            f.write(dict_str)

    def train_model(self, max_depth: int = None, gain_type: str = 'IG', directory: str = None, attributes: dict = None) -> None:
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

    def run_ID3(self, s : list, attributes : dict, labels : list, current_depth : int, depth_limit: int = None) -> dict:
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
        # if out of attributes
        if not attributes:
            counts = Counter(s_labels)
            return counts.most_common(1)[0][0]
        "TODO: finish"