#!/usr/bin/env python3

import numpy as np

class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification."""

    def __init__(self, nx, nodes):
        """Initializes the neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        
        # Initialize weights and biases for the hidden and output layer
        self.W1 = np.random.randn(nodes, nx)  # Random initialization for hidden layer
        self.b1 = np.zeros((nodes, 1))        # Bias for hidden layer (zeros)
        self.A1 = np.zeros((nodes, 1))        # Activated output for hidden layer (zeros)
        self.W2 = np.random.randn(1, nodes)   # Random initialization for output layer
        self.b2 = np.zeros((1, 1))            # Bias for output layer (zero)
        self.A2 = np.zeros((1, 1))            # Activated output for output layer (zeros)
