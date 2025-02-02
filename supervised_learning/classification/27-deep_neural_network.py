#!/usr/bin/env python3
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Deep neural network class for multiclass classification."""
    
    def __init__(self, nx, layers):
        """Initialize the deep neural network."""
        if not isinstance(nx, int) or nx < 1:
            raise TypeError("nx must be a positive integer")
        if (not isinstance(layers, list) or len(layers) < 1
                or not all(isinstance(l, int) and l > 0 for l in layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        for i in range(self.L):
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
            if i == 0:
                self.weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                )
            else:
                self.weights['W' + str(i + 1)] = (
                    np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                )
    
    def forward_prop(self, X):
        """Perform forward propagation."""
        self.cache['A0'] = X
        for i in range(1, self.L):
            Z = (self.weights['W' + str(i)] @ self.cache['A' + str(i - 1)] +
                 self.weights['b' + str(i)])
            self.cache['A' + str(i)] = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        
        ZL = (self.weights['W' + str(self.L)] @ self.cache['A' + str(self.L - 1)] +
              self.weights['b' + str(self.L)])
        self.cache['A' + str(self.L)] = np.exp(ZL) / np.sum(np.exp(ZL), axis=0, keepdims=True)  # Softmax
        
        return self.cache['A' + str(self.L)], self.cache
    
    def cost(self, Y, A):
        """Compute cost using cross-entropy loss."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m
    
    def evaluate(self, X, Y):
        """Evaluate the model."""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == labels)
        return predictions, self.cost(Y, A)
    
    def gradient_descent(self, Y, alpha=0.05):
        """Perform one iteration of gradient descent."""
        m = Y.shape[1]
        dZL = self.cache['A' + str(self.L)] - Y
        
        for i in reversed(range(1, self.L + 1)):
            dW = (dZL @ self.cache['A' + str(i - 1)].T) / m
            db = np.sum(dZL, axis=1, keepdims=True) / m
            
            if i > 1:
                dZL = (self.weights['W' + str(i)].T @ dZL) * (self.cache['A' + str(i - 1)] * (1 - self.cache['A' + str(i - 1)]))
            
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db
    
    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the deep neural network."""
        if not isinstance(iterations, int) or iterations <= 0:
            raise TypeError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise TypeError("alpha must be positive")
        
        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, alpha)
        
        return self.evaluate(X, Y)
    
    def save(self, filename):
        """Save the instance to a file."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """Load an instance from a file."""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
