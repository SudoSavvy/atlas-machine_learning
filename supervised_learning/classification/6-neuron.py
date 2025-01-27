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
        
        self.__W = np.random.randn(1, nx)
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
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Compute the cost using binary cross-entropy."""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one pass of gradient descent to update weights and bias."""
        m = X.shape[1]
        dZ = A - Y
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m
        
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def evaluate(self, X, Y):
        """Evaluate the neuron’s predictions."""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)
        return prediction, cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neuron."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
