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
        if (not isinstance(layers, list) or len(layers) == 0 or 
            any(not isinstance(x, int) or x <= 0 for x in layers)):
            raise TypeError("layers must be a list of positive integers")
        
        self.__L = len(layers)
        self.cache = {}
        self.weights = {}
        
        for l in range(1, self.__L + 1):
            if l == 1:
                self.weights[f'W{l}'] = np.random.randn(layers[l - 1], nx) * np.sqrt(2 / nx)
            else:
                self.weights[f'W{l}'] = np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2 / layers[l - 2])
            self.weights[f'b{l}'] = np.zeros((layers[l - 1], 1))
    
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
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost
    
    def forward_prop(self, X):
        """
        Placeholder forward propagation method. Should be implemented properly.
        """
        return np.random.rand(1, X.shape[1])  # Dummy output for evaluation testing
