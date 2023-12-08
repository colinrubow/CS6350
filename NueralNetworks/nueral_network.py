import numpy as np

class NueralNetwork():
    # A class implementing a nueral network.
    def __init__(self, num_inputs: int) -> None:
        # the network will be oraganized as follows:
        # a list of layers
        # a layer is a list of nodes (the first layer is just values representing the input and has no weights)
        # a node is a list being [node_value, weights]
        # a node_value is a float, the value of the node
        # a weight is a list of lists where the values of the lists are [node_index, weight value, partial]
        #   since some nodes don't have weights associated with them or aren't fully connected.
        #   the partial is for back propogation
        # overall: network[layers[nodes[node_value, weights[[node_index, weight_value, partial]]]]]
        self.network = [[[1] for _ in range(num_inputs)]]
        
    def add_layer(self, layer_number: int = -1, num_nodes: int = 1) -> None:
        # a method for adding a layer. If the layer is the current one, use layer_number -1.
        # num_nodes is the number of nodes in that layer.
        # add layer
        if layer_number == -1:
            self.network.append([[1, [[] for _ in range(len(self.network[(layer_number - 1)%len(self.network)]))]] for _ in range(num_nodes)])
        else:
            self.network.insert(layer_number, [[1, [[] for _ in range(len(self.network[(layer_number - 1)%len(self.network)]))]] for _ in range(num_nodes)])
        # modify the next layer if exists
        if len(self.network) - 1 > layer_number%len(self.network):
            self.network[layer_number + 1] = [[1, [[] for _ in range(num_nodes)]] for _ in range(len(self.network[layer_number + 1]))]

    def add_weights(self, self_initialize: str = 'gaussian', weights: list[tuple[int, int, list[list[int, float, float]]]] = None) -> None:
        # a method for adding weights. Self_initialize can be 'gaussian' or 'zero' which defines the hard coded weight initialization
        # strategy. self_initialization is only applicable if weights is None. The weights are in the format given where the first int is the layer and the second int is the node number
        # for where you want to add weights to. The first int in the nested lists is the node that weight is multiplied with
        # the first float is the value of the weight, and the second float is the value of the partial derivative used for 
        # back propogation. It can be initailized to anything.
        # the weights are lists where the first value is the layer, the second is the node, the third is the list of weights
        # don't forget to add a third dummy value to the list for back propogation caching
        if weights is not None:
            for weight in weights:
                self.network[weight[0]][weight[1]][1] = weight[2]
        else:
            match self_initialize:
                case 'gaussian':
                    weight_function = np.random.normal
                case 'zero':
                    weight_function = self.__zero
                case _:
                    raise Exception("not valid self_initialize value")
            for i, layer in enumerate(self.network):
                if i == 0:
                    continue
                for j, node in enumerate(layer):
                    # if j == 0 then no weights (bias node) unless its the output node
                    if j == 0 and i != len(self.network) - 1:
                        continue
                    for k, weight in enumerate(node):
                        # all weight should be [], initially
                        if not weight:
                            raise Exception('a weight is not [] as supposed')
                        self.network[i][j][1][k] = [k, weight_function(), 1]
    
    def forward(self, input: list[float]) -> np.ndarray:
        # runs the forward algorithm of the network. All layers are sigmoid activation except the last which is linear
        # set the first layer
        self.network[0] = input
        # iterate till the last layer
        for i, layer in enumerate(self.network[1:-1]):
            for node in layer:
                # check if there are any weights in this node
                if any(node[1]):
                    node[0] = self.__sigmoid(sum([self.network[i][weight[0]][0]*weight[1] for weight in node[1] if weight]))
                else:
                    continue
        # last layer
        for node in self.network[-1]:
            node[0] = sum([self.network[-2][weight[0]][0]*weight[1] for weight in node[1] if weight])
        output = np.array([node[0] for node in self.network[-1]])
        return output
    
    def backwards(self, y_star: float, y: float) -> np.ndarray:
        # runs the backwards propogation algorithm. y_star is the forward output, y is the desired output.
        # initialize gradient
        gradient = [np.nan]*sum([len(node[1]) for layer in self.network[1:] for node in layer])
        # iterate from last layer to second to last, top weights to bottom weights
        # compute partial of L with respect to y
        L_partial_y = y - y_star
        for i, layer in enumerate(reversed(self.network)):
            i_forward = len(self.network) - 1 - i
            # if on first layer
            if i_forward == 0:
                continue
            for j, node in enumerate(layer):
                for k, weight in enumerate(node[1]):
                    if not weight:
                        continue
                    # find all paths from weight[0] to y
                    # will be of format [next_layer(node_index, weight_index), ...]
                    paths = [(j, k), []]
                    paths = self.__find_path(paths, i_forward+1)
                    # calculate the top partial
                    if i == 0:
                        temp = 1
                    else:
                        temp = node[0]*(1 - node[0])
                    weight[2] = temp*weight[1]
                    # multiply down each path and add together
                    total = self.__multiply_path(0, temp*self.network[i_forward - 1][weight[0]][0], paths, i_forward, True)
                    gradient[gradient.index(np.nan)] = total*L_partial_y
        return np.array(gradient[:gradient.index(np.nan)])

    def train(self, train_data: np.ndarray, test_data: np.ndarray, initial_gamma: float = 1, d: float = 1, convergence_limit: float = 1e-5) -> tuple[list, tuple[float, float]]:
        # to train the network. Train_data and test_data is the normal format. Initial_gamma is the initial learning rate. d is the division factor.
        # convergence_limit is the size of the gradient we want it to get to.
        train_error = []
        for epoch in range(500):
            rate = initial_gamma/(1 + initial_gamma/d*epoch)
            train_error.append(self.test(train_data))
            np.random.shuffle(train_data)
            for example in train_data:
                y = example[-1]
                x = example[0:-1]
                x = np.array([[ex] for ex in x])
                y_star = self.forward(x)[0]
                gradient = self.backwards(y_star, y)
                if np.linalg.norm(gradient) < convergence_limit:
                    print('converged!')
                    return (train_error, (self.test(train_data), self.test(test_data)))
                # update the weights
                gradient_index = 0
                for i, layer in enumerate(reversed(self.network)):
                    i_forward = len(self.network) - 1 - i
                    if i_forward == 0:
                        continue
                    for j, node in enumerate(layer):
                        for k, weight in enumerate(node[1]):
                            if not weight:
                                continue
                            self.network[i_forward][j][1][k][1] = self.network[i_forward][j][1][k][1] + rate*gradient[gradient_index]
                            gradient_index += 1
        print('oh no, didn\'t converge!')
        return (train_error, (self.test(train_data), self.test(test_data)))

    def test(self, data):
        # given some data, returns the percent error for the network
        num_incorrect = 0
        for example in data:
            y = example[-1]
            x = example[0:-1]
            x = [[ex] for ex in x]
            y_star = self.forward(x)[0]
            if np.sign(y) != np.sign(y_star):
                num_incorrect += 1
        return num_incorrect/len(data)

    def __multiply_path(self, total: float, partial: float, paths: list, layer: int, first: bool):
        # recursive function for back propogation
        if not first:
            partial *= self.network[layer][paths[0][0]][1][paths[0][1]][2]
        # check if last layer
        if layer == len(self.network) - 1:
            total += partial
            return total
        for p in paths[1]:
            total = self.__multiply_path(total, partial, p, layer+1, False)
        return total

    def __find_path(self, path: list[tuple], layer: int):
        # recursive function for back propogation
        if layer > len(self.network)-1:
            return path
        path_append = []
        for n, node in enumerate(self.network[layer]):
            for w, weight in enumerate(node[1]):
                if not weight:
                    continue
                if weight[0] == path[0][0]:
                    new_path = [(n, w), []]
                    path_append.append(new_path)
                    if layer + 1 <= len(self.network) - 1:
                        self.__find_path(new_path, layer + 1)
        path[1] = path_append
        return path

    @staticmethod
    def __sigmoid(x: float):
        return 1/(1 + np.e**(-x))

    @staticmethod
    def __zero():
        return 0