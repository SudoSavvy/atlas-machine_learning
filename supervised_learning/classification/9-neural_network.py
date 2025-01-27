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
