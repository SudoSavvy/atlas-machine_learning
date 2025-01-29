#!/usr/bin/env python3
"""Deep Neural Network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        Initializes a deep neural network.

        Parameters:
        nx (int): Number of input features.
        layers (list): Number of nodes in each layer of the network.

        Raises:
        TypeError: If nx is not an integer.
        ValueError: If nx is less than 1.
        TypeError: If layers is not a list or is an empty list.
        TypeError: If any element in layers is not a positive integer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(n, int) or n <= 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # Number of layers
        self.cache = {}  # Dictionary to store intermediate values
        self.weights = {}  # Dictionary to store weights and biases

        # Initialize weights and biases using He et al. method
        prev_nodes = nx
        for l in range(1, self.L + 1):
            self.weights[f"W{l}"] = np.random.randn(layers[l - 1], prev_nodes) * np.sqrt(2 / prev_nodes)
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))
            prev_nodes = layers[l - 1]
