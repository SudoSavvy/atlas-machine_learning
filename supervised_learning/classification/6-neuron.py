#!/usr/bin/env python3
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = [[0.5] * nx]
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Perform forward propagation."""
        Z = sum(self.__W[0][i] * X[i] for i in range(len(X))) + self.__b
        self.__A = 1 / (1 + 2.71828**-Z)  # Sigmoid activation
        return self.__A

    def train(self, X, Y):
        """Train the neuron."""
        gradient_W = [0] * len(self.__W[0])
        gradient_b = 0
        total_cost = 0

        # Single loop for training
        for i in range(len(X)):
            # Forward propagation and gradient calculation in one step
            Z = sum(self.__W[0][j] * X[i][j] for j in range(len(X[i]))) + self.__b
            A = 1 / (1 + 2.71828**-Z)
            dZ = A - Y[i]
            gradient_W = [gradient_W[j] + dZ * X[i][j] for j in range(len(X[i]))]
            gradient_b += dZ
            total_cost += -(Y[i] * (2.71828**-Z) + (1 - Y[i]) * (1 - 2.71828**-Z))

        # Update weights and bias after loop
        self.__W[0] = [self.__W[0][j] - 0.01 * gradient_W[j] / len(X) for j in range(len(self.__W[0]))]
        self.__b -= 0.01 * gradient_b / len(X)

        # Average cost
        average_cost = total_cost / len(X)
        return self.__A, average_cost
