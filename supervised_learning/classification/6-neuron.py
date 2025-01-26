#!/usr/bin/env python3
import numpy as np


class Neuron:
    def __init__(self, nx):
        # Initialize the neuron with random weights and zero bias
        self.__W = np.random.randn(1, nx)  # Weights of the neuron (1, nx)
        self.__b = 0  # Bias initialized to zero
        self.__A = 0  # Activation output initialized to zero

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
        # Perform forward propagation to get the activation output A
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        return self.__A

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

    def cost(self, Y, A):
        # Compute the cost using binary cross-entropy
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        # Check if iterations is a positive integer
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # Check if alpha is a positive float
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Train the model over the specified number of iterations (Single loop for both forward_prop and gradient descent)
        for i in range(iterations):
            # Perform forward propagation
            A = self.forward_prop(X)

            # Compute the cost
            cost = self.cost(Y, A)

            # Perform one step of gradient descent
            self.gradient_descent(X, Y, A, alpha)

        # Return the final activation output and the cost after training
        return A, cost

    def evaluate(self, X, Y):
        # Perform forward propagation to get the final activated output
        A = self.forward_prop(X)

        # Compute the cost
        cost = self.cost(Y, A)

        # Convert the activation output to binary predictions (0 or 1)
        prediction = (A >= 0.5).astype(int)

        return prediction, cost
