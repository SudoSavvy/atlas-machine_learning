#!/usr/bin/env python3
"""Class DeepNeuralNetwork that defines a deep neural
network performing binary classification based on
26-deep_neural_network.py"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """Class DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # setting up the private instance attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # initializing all weights and biases of the network
        prev_nodes = nx
        for i, nodes in enumerate(layers, 1):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            # setting variable key to the current layer index
            key = f'{i}'
            # using the He et al. method to initialize weights
            self.weights[f'W{key}'] = np.random.randn(nodes, prev_nodes
                                                      ) * np.sqrt(
                                                          2 / prev_nodes)
            # initialized to 0s and saved the weights dictionary
            self.weights[f'b{key}'] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """use getter method for L"""
        return self.__L

    @property
    def cache(self):
        """use getter method for cache"""
        return self.__cache

    @property
    def weights(self):
        """use getter method for weights"""
        return self.__weights
    
    def forward_prop(self, X):
        """calculates the forward propagation of the neural
        network"""
        # using the sigmoid function
        def sig(Z):
            """sigmoid function"""
            return 1 / (1 + np.exp(-Z))

        # X saved to the cache dictionary using key A0
        self.__cache['A0'] = X

        # iterating each layer by applying W, b, Z
        A = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            Z = np.dot(W, A) + b
            A = sig(Z)
            self.__cache[f'A{i}'] = A

        # returning output and the cache that was updated
        return A, self.__cache

    def cost(self, Y, A):
        """calculates the cost of the model using a logistic
        regression"""
        log_A = np.log(A)
        log_1_minus_A = np.log(1.0000001 - A)
        loss = Y * log_A + (1 - Y) * log_1_minus_A

        # calculate the average cost
        return -np.sum(loss) / A.shape[1]

    def evaluate(self, X, Y):
        """evaluates the neural network's predictions and returns
        the neuron's prediction and the cost of the network"""
        A, _ = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calculates one pass of gradient descent on the neural
        network

        Y = correct labels
        cache = dictionary containing all the intermediary values
        alpha = learning rate

        """
        # fetching the training examples m and the number of layers L
        m = Y.shape[1]
        L = self.__L

        # loop back performing the backpropagation
        for i in range(L, 0, -1):
            # checks if the layer is the outer layer L, if so dz is dA
            if i == L:
                dA = cache[f'A{L}'] - Y
                dz = dA
            else:
                # if it isn't the outer layer, the hidden layer, compute dz
                A = cache[f'A{i}']
                dz = dA * A * (1 - A)

            # calculates the gradient weights and biases then get avg
            dw = np.matmul(dz, cache[f'A{i-1}'].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            # calculate and backpropagate the error in upcoming layer
            dA = np.matmul(self.__weights[f'W{i}'].T, dz)

            # updating weights and biases of the current layer
            self.__weights[f'W{i}'] -= alpha * dw
            self.__weights[f'b{i}'] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """trains the deep neural network and returns the
        evaluation of the training data after iterations of
        training have occurred"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # storing costs and iterations to a list for the graph
        if graph:
            g_costs = []
            g_iters = []

        # using forward propagation and gradient descent
        for iteration in range(iterations + 1):
            # calling forward prop and returning values
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

            # calculates the current cost and print at the set interval
            if verbose and iteration % step == 0:
                cost = self.cost(Y, A)
                print(f"Cost after {iteration} iterations: {cost}")

            # appending cost and iteration to the list
            if graph and iteration % step == 0:
                cost = self.cost(Y, A)
                g_costs.append(cost)
                g_iters.append(iteration)

        # plotting the graph and then displaying
        if graph:
            plt.plot(g_costs, g_iters, color='blue')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()

        # returning the evaluation of the training data
        return self.evaluate(X, Y)

    def save(self, filename):
        """saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as pf:
            pickle.dump(self, pf)

    def load(filename):
        """loads a pickled DeepNeuralNetwork object"""
        if not filename.endswith('pkl'):
            filename += 'pkl'
        try:
            with open(filename, 'rb') as pf:
                return pickle.load(pf)
        except FileNotFoundError:
            return None