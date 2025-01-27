#!/usr/bin/env python3
import numpy as np

class NeuralNetwork:
    def __init__(self, nx, nh):
        """
        Initializes the neural network with one hidden layer.
        
        nx: The number of input features
        nh: The number of hidden units
        """
        if not isinstance(nx, int) or nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nh, int) or nh <= 0:
            raise ValueError("nh must be a positive integer")
        
        self.__W1 = np.random.randn(nh, nx) * 0.01  # Weights  hidden layer
        self.__b1 = np.zeros((nh, 1))  # Bias  hidden layer
        self.__A1 = np.zeros((nh, 1))  # Activation  hidden layer
        self.__W2 = np.random.randn(1, nh) * 0.01  # Weights  output layer
        self.__b2 = np.zeros((1, 1))  # Bias  output layer
        self.__A2 = np.zeros((1, 1))  # Activation  output layer

    def sigmoid(self, Z):
        """
        Sigmoid activation function.
        
        Z: Input to the sigmoid function
        """
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        """
        Performs forward propagation of the neural network.
        
        X: Input data
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A2

    def cost(self, Y, A2):
        """
        Computes the cost function.
        
        Y: True labels
        A2: Predicted labels
        """
        m = Y.shape[1]
        return -1/m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))

    def backward_propagation(self, X, Y):
        """
        Performs backward propagation to compute gradients.
        
        X: Input data
        Y: True labels
        """
        m = X.shape[1]
        dZ2 = self.__A2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.__A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.__W2.T, dZ2) * self.__A1 * (1 - self.__A1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        gradients = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }
        return gradients

    def update_parameters(self, gradients, alpha):
        """
        Updates the weights and biases of the network using gradient descent.
        
        gradients: Gradients from backward propagation
        alpha: Learning rate
        """
        self.__W1 -= alpha * gradients["dW1"]
        self.__b1 -= alpha * gradients["db1"]
        self.__W2 -= alpha * gradients["dW2"]
        self.__b2 -= alpha * gradients["db2"]

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network.
        
        X: Input data (nx, m)
        Y: True labels (1, m)
        iterations: Number of iterations
        alpha: Learning rate
        """
        # Validation of inputs
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Training loop
         _ in range(iterations):
            # Forward propagation
            A2 = self.forward_propagation(X)
            
            # Compute cost
            cost = self.cost(Y, A2)
            
            # Backward propagation
            gradients = self.backward_propagation(X, Y)
            
            # Update parameters
            self.update_parameters(gradients, alpha)

        # Return the final cost after training
        return cost
