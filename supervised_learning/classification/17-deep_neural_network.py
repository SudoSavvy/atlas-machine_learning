#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    """
    DeepNeuralNetwork class defines a deep neural network performing binary classification.
    
    Attributes:
        L (int): Number of layers in the network.
        cache (dict): A dictionary that stores intermediary values (e.g., activations) each layer.
        weights (dict): A dictionary that stores weights and biases each layer.
    
    Methods:
        __init__(self, nx, layers): Initializes the deep neural network with given parameters.
        get_L(self): Getter method the number of layers.
        getcache(self): Getter method the cache.
        getweights(self): Getter method the weights.
    """
    
    def __init__(self, nx, layers):
        """
        Initializes the DeepNeuralNetwork with the number of input features and layers.
        
        Args:
            nx (int): The number of input features.
            layers (list): A list representing the number of nodes in each layer of the network.
        
        Raises:
            TypeError: If nx is not an integer or if layers is not a list of positive integers.
            ValueError: If nx is less than 1 or if any layer in layers is not a positive integer.
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate layers
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        # Check if all elements are positive integers
        if (not isinstance(layers, list) or len(layers) == 0 or 
            list(map(lambda x: isinstance(x, int) and x > 0, layers)).count(False) > 0):
            raise TypeError("layers must be a list of positive integers")


        # Initialize private attributes
        self.__L = len(layers)  # Number of layers
        self.__cache = {}       # Cache activations
        self.__weights = {}     # Weights and biases dictionary
        
        # Initialize weights and biases each layer
        for l in range(1, self.__L + 1):
            # He initialization weights (Wl)
            if l == 1:
                self.weights[f'W{l}'] = np.random.randn(layers[l - 1], nx) * np.sqrt(2 / nx)
            else:
                self.weights[f'W{l}'] = np.random.randn(layers[l - 1], layers[l - 2]) * np.sqrt(2 / layers[l - 2])
            
            # Biases initialized to 0
            self.weights[f'b{l}'] = np.zeros((layers[l - 1], 1))

    def get_L(self):
        """Getter method the number of layers."""
        return self.__L

    def get_cache(self):
        """Getter method the cache."""
        return self.__cache

    def get_weights(self):
        """Getter method the weights and biases."""
        return self.__weights
