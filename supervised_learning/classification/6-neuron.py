#!/usr/bin/env python3
import numpy as np


class Neuron:
    def __init__(self, nx):
        # Initialize the neuron with random weights and zero bias
        self.__W = np.random.randn(1, nx)
        self.__b = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    def forward_prop(self, X):
        # Perform forward propagation to get the activation output A
        Z = np.dot(self.__W, X) + self.__b
        A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        return A

    def gradient_descent(self, X, Y, A, alpha=0.05):
        # Calculate the number of examples (m)
        m = X.shape[1]

        # Compute the gradients of the cost function with respect to W and b
        dZ = A - Y  # The derivative of the sigmoid cost function
        dW = np.dot(dZ, X.T) / m  # Gradient with respect to W
        db = np.sum(dZ) / m  # Gradient with respect to b

        # Update the weights and bias using the learning rate alpha
        self.__W -= alpha * dW
        self.__b -= alpha * db
