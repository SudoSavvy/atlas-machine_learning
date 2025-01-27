#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer for binary classification.
"""

import numpy as np


class NeuralNetwork:
    """
    A neural network with one hidden layer performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.

        Parameters:
        nx (int): Number of input features.
        nodes (int): Number of nodes in the hidden layer.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        TypeError: If nodes is not an integer.
        ValueError: If nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer
        self.__W1 = np.random.randn(nodes, nx)  # Weight matrix for hidden layer
        self.__b1 = np.zeros((nodes, 1))  # Bias vector for hidden layer
        self.__A1 = 0  # Activated output for hidden layer

        # Output neuron
        self.__W2 = np.random.randn(1, nodes)  # Weight vector for output neuron
        self.__b2 = 0  # Bias for output neuron
        self.__A2 = 0  # Activated output for output neuron (prediction)

    @property
    def W1(self):
        """Getter for W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for A2."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m),
                           where nx is the number of features and m is the number of examples.

        Returns:
        tuple: Activated outputs for the hidden layer (__A1) and the output neuron (__A2).
        """
        # Calculate the activated output for the hidden layer
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation

        # Calculate the activated output for the output neuron
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
        Y (numpy.ndarray): Correct labels for the input data with shape (1, m).
        A (numpy.ndarray): Activated output of the neuron for each example with shape (1, m).

        Returns:
        float: The cost of the model.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Parameters:
        X (numpy.ndarray): Input data with shape (nx, m),
                           where nx is the number of features and m is the number of examples.
        Y (numpy.ndarray): Correct labels for the input data with shape (1, m).

        Returns:
        tuple: The prediction and the cost of the network.
            - Prediction is a numpy.ndarray with shape (1, m), containing predicted labels (0 or 1).
            - Cost is a float representing the cost of the network.
        """
        # Perform forward propagation
        self.forward_prop(X)

        # Calculate the cost
        cost = self.cost(Y, self.__A2)

        # Determine predictions (0 or 1)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)

        return prediction, cost
