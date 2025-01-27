#!/usr/bin/env python3
"""
Defines a neural network with one hidden layer  binary classification.
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
        self.W1 = np.random.randn(nodes, nx)  # Weight matrix  hidden layer
        self.b1 = np.zeros((nodes, 1))  # Bias vector  hidden layer
        self.A1 = 0  # Activated output  hidden layer

        # Output neuron
        self.W2 = np.random.randn(1, nodes)  # Weight vector  output neuron
        self.b2 = 0  # Bias  output neuron
        self.A2 = 0  # Activated output  output neuron (prediction)
