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
        self.W1 = np.random.randn(nodes, nx)  # Random initialization for the hidden layer
        self.b1 = np.zeros((nodes, 1))        # Bias for the hidden layer (zeros)
        self.A1 = np.zeros((nodes, 1))        # Activated output for the hidden layer (zeros)
        self.W2 = np.random.randn(1, nodes)   # Random initialization for the output layer
        self.b2 = np.zeros((1, 1))            # Bias for the output layer (zero)
        self.A2 = np.zeros((1, 1))            # Activated output for the output layer (zeros)

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def forward_prop(self, X):
        """Performs forward propagation."""
        Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(Z1)
        
        Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(Z2)
        
        return self.A1, self.A2

# Example usage
nx = 5  # Number of input features
nodes = 3  # Number of nodes in the hidden layer
X = np.random.randn(nx, 10)  # 10 examples with nx features each

# Initialize the neural network
nn = NeuralNetwork(nx, nodes)

# Perform forward propagation
A1, A2 = nn.forward_prop(X)

# Print the activated output of the second layer (binary classification output)
print("Output of the network: \n", A2)
