#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification."""

    def __init__(self, nx, nodes):
        """
        Initializes the neural network.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases  the hidden and output layers
        self.W1 = np.random.randn(nodes, nx)  # Hidden layer weights
        self.b1 = np.zeros((nodes, 1))        # Hidden layer biases
        self.A1 = np.zeros((nodes, 1))        # Hidden layer activated output
        self.W2 = np.random.randn(1, nodes)  # Output layer weights
        self.b2 = np.zeros((1, 1))           # Output layer bias
        self.A2 = np.zeros((1, 1))           # Output layer activated output
