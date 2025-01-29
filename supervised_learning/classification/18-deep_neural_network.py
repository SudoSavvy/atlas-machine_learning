#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network for binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the Deep Neural Network.
        
        nx (int): The number of input features.
        layers (list): A list representing the number of neurons in each layer.
        """
        self.__L = len(layers)  # Number of layers
        self.__cache = {}  # Cache dictionary
        self.__weights = {}  # Weights dictionary
        self.__initialize_weights(layers)

    def __initialize_weights(self, layers):
        """
        Initializes the weights and biases for each layer of the neural network.
        
        layers (list): A list representing the number of neurons in each layer.
        """
        for l in range(1, self.__L):
            self.__weights["W" + str(l)] = np.random.randn(layers[l], layers[l - 1]) * 0.01
            self.__weights["b" + str(l)] = np.zeros((layers[l], 1))

    def sigmoid(self, Z):
        """
        Sigmoid activation function.
        
        Z (ndarray): The input to the sigmoid function.
        
        Returns:
        ndarray: The result of applying the sigmoid activation.
        """
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        
        X (ndarray): The input data, with shape (nx, m), where:
            - nx is the number of input features
            - m is the number of examples
        
        Returns:
        A (ndarray): The output of the neural network.
        cache (dict): A dictionary with the activated outputs of each layer.
        """
        self.__cache["A0"] = X  # Save the input data in the cache
        
        A = X
        for l in range(1, self.__L):
            Z = np.dot(self.__weights["W" + str(l)], A) + self.__weights["b" + str(l)]
            A = self.sigmoid(Z)  # Apply sigmoid activation
            self.__cache["A" + str(l)] = A  # Save activated output to the cache

        return A, self.__cache
