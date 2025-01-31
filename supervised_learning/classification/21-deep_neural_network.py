#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the DeepNeuralNetwork.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of positive integers.
            ValueError: If nx is less than 1 or any layer in layers is not a positive integer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.__L = len(layers)  # Number of layers
        self.__cache = {}  # Stores intermediate values (Z, A) during forward propagation
        self.__weights = {}  # Stores weights and biases

        # Initialize weights and biases using He initialization (1 loop)
        for l in range(1, self.__L + 1):
            if l == 1:
                self.__weights[f'W{l}'] = np.random.randn(layers[l - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[f'W{l}'] = np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2 / layers[l - 2])
            self.__weights[f'b{l}'] = np.zeros((layers[l - 1], 1))

    @property
    def L(self):
        """Getter the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter the cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter the weights."""
        return self.__weights

    def forward_prop(self, X):
        """
        Performs forward propagation on the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: (A, cache), where A is the output of the last layer and cache contains all intermediate values.
        """
        self.__cache['A0'] = X  # Input layer
        # Forward propagation (1 loop)
        for l in range(1, self.__L + 1):
            Z = np.dot(self.__weights[f'W{l}'], self.__cache[f'A{l-1}']) + self.__weights[f'b{l}']
            A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f'A{l}'] = A
            self.__cache[f'Z{l}'] = Z
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """
        Computes the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output of the neuron with shape (1, m).

        Returns:
            float: Cost value.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            tuple: (Predictions, Cost value)
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            cache (dict): Dictionary containing all the intermediary values of the network.
            alpha (float): Learning rate.

        Updates:
            The private attribute __weights.
        """
        m = Y.shape[1]
        dZ = cache[f'A{self.__L}'] - Y  # Derivative of the cost with respect to Z in the output layer
        # Backpropagation (1 loop)
        for l in range(self.__L, 0, -1):
            A_prev = cache[f'A{l-1}']
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if l > 1:
                dZ = np.dot(self.__weights[f'W{l}'].T, dZ) * (A_prev * (1 - A_prev))  # Derivative previous layer
            self.__weights[f'W{l}'] -= alpha * dW
            self.__weights[f'b{l}'] -= alpha * db