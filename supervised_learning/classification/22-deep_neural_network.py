#!/usr/bin/env python3
""" Module creating a class NeuronalNetwork"""
import numpy as np


class DeepNeuralNetwork():
    """ Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initiates  Deep Neural Network class

        Inputs:
        nx - number of input features
            * must be integer of value greater than or equal to 1
        layers - number of nodes found in the each layer of the network
            * must be a list of positive integers

        Public Instance Attributes:
        L - The number of layers in the neural network
        cache - A dictionary holding all intermediary values of the network.
            Empty on instantiation
        weights - Dictionary holding all weights and biases of the network
        """

        nx_is_int = isinstance(nx, int)
        nx_ge_1 = nx >= 1
        layers_is_list_ints = isinstance(layers, list)

        if not nx_is_int:
            raise TypeError("nx must be an integer")
        if not nx_ge_1:
            raise ValueError("nx must be a positive integer")
        if not layers_is_list_ints:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        # Private Properties
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        previous = nx
        weights_dict = {}

        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] < 0:
                raise TypeError("layers must be a list of positive integers")

            weights_dict["W{}".format(l + 1)] = (np.random.randn(layers[l],
                                                                 previous) *
                                                 np.sqrt(2 / previous))
            weights_dict["b{}".format(l + 1)] = np.zeros((layers[l], 1))
            previous = layers[l]

        self.__weights = weights_dict

    @property
    def L(self):
        """Getter for the private L attribute"""
        return self.__L

    @property
    def weights(self):
        """Getter for the private weights attribute"""
        return self.__weights

    @property
    def cache(self):
        """Getter for the private cache attribute"""
        return self.__cache

    def forward_prop(self, X):
        """
        Calculates the forward propogation of the neural network. All neurons
        will use the sigmoid activation function.

        Inputs:
        X - a numpy.ndarray that contains the input data.

        Updates:
        __cache as a ditionary with the output of each layer as A{l}

        Returns:
        Returns the output of the neural network"""
        self.__cache["A0"] = X
        for layer in range(self.L):
            W = self.weights["W{}".format(layer + 1)]
            b = self.weights["b{}".format(layer + 1)]
            current_A = self.cache["A{}".format(layer)]
            z = np.matmul(W, current_A) + b
            A = 1 / (1 + (np.exp(-z)))
            self.__cache["A{}".format(layer + 1)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        C=-1/m(∑(Y⋅log(A)+((1-Y)⋅log(1-A))))
        where m is the number of training examples

        Inputs:
        Y represents the correct labels for input data,
        A represents the activated output of the neuron for each example

        Returns:
        C - Cost of the model
        """
        m = Y.shape[1]
        C = -1 / m * (np.sum((Y * np.log(A)) + ((1 - Y) *
                                                np.log(1.0000001 - A))))
        return C

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.
        Prediction is forward propagation evaluated to a 1 or a 0.

        Inputs:
        X - numpy.ndarray which contains the input data
        Y - numpy.ndarray which contains the correct labels for the input data

        Returns:
        Returns the prediction and the cost of the network a tuple.
        """

        A, B = self.forward_prop(X)
        predict = np.where(A >= 0.5, 1, 0)

        C = self.cost(Y, A)
        return (predict, C)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Inputs:
        Y - numpy.ndarray with correct labels for the input data
        cache - dictionary containing all the intermediary
            values of the network
        alpha - the learning rate

        Updates:
        __weights
        """

        m = Y.shape[1]

        for layer in range(self.L, 0, -1):
            A_current = self.cache["A{}".format(layer)]
            A_previous = self.cache["A{}".format(layer - 1)]

            if layer == self.__L:
                dz = (A_current - Y)
            else:
                dz = dA_prev * (A_current * (1 - A_current))

            dW = (1 / m) * (np.matmul(dz, A_previous.T))
            db = (1 / m) * (np.sum(dz, axis=1, keepdims=True))

            W = self.weights["W{}".format(layer)]
            dA_prev = np.matmul(W.T, dz)

            self.__weights["W{}".format(layer)] = (
                self.__weights["W{}".format(layer)] - (alpha * dW))
            self.__weights["b{}".format(layer)] = (
                self.__weights["b{}".format(layer)] - (alpha * db))

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron

        Inputs:
        X - numpy.ndarray which contains the input data
        Y - numpy.ndarray which contains the correct labels for the input data
        iterations - number of iterations to train over
            * Must be a positive integer
        alpha - learning rate
            * Must be a positive float value

        Return:
        Returns the evaluation of the training data after all iterations
        """

        iter_is_int = isinstance(iterations, int)
        iter_is_pos = iterations > 0
        alpha_is_float = isinstance(alpha, float)
        alpha_is_pos = alpha > 0

        if not iter_is_int:
            raise TypeError("iterations must be an integer")
        if not iter_is_pos:
            raise ValueError("iterations must be a positive integer")
        if not alpha_is_float:
            raise TypeError("alpha must be a float")
        if not alpha_is_pos:
            raise ValueError("alpha must be positive")

        for i in range(0, iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)

        return self.evaluate(X, Y)