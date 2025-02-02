#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if type(nx) is not int or nx < 1:
            raise TypeError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        for l in range(self.L):
            if type(layers[l]) is not int or layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            
            key_W = "W" + str(l + 1)
            key_b = "b" + str(l + 1)
            
            if l == 0:
                self.weights[key_W] = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)
            else:
                self.weights[key_W] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
            
            self.weights[key_b] = np.zeros((layers[l], 1))
    
    def forward_prop(self, X):
        self.cache["A0"] = X
        
        for l in range(1, self.L + 1):
            W = self.weights["W" + str(l)]
            b = self.weights["b" + str(l)]
            A_prev = self.cache["A" + str(l - 1)]
            Z = np.matmul(W, A_prev) + b
            
            if l == self.L:
                A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)  # Softmax
            else:
                A = 1 / (1 + np.exp(-Z))  # Sigmoid
            
            self.cache["A" + str(l)] = A
        
        return A, self.cache
    
    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m  # Cross-entropy loss
    
    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        cost = self.cost(Y, A)
        return predictions, cost
    
    def gradient_descent(self, Y, alpha=0.05):
        m = Y.shape[1]
        A_L = self.cache["A" + str(self.L)]
        dZ = A_L - Y  # Derivative of cross-entropy loss with softmax
        
        for l in reversed(range(1, self.L + 1)):
            A_prev = self.cache["A" + str(l - 1)]
            W = self.weights["W" + str(l)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))  # Sigmoid derivative
            
            self.weights["W" + str(l)] -= alpha * dW
            self.weights["b" + str(l)] -= alpha * db
    
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True):
        if type(iterations) is not int or iterations <= 0:
            raise TypeError("iterations must be a positive integer")
        if type(alpha) is not float or alpha <= 0:
            raise TypeError("alpha must be a positive float")
        
        costs = []
        
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, alpha)
            
            if verbose and i % 100 == 0:
                cost = self.cost(Y, self.cache["A" + str(self.L)])
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
        
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(range(0, iterations, 100), costs, 'b-')
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()
        
        return self.evaluate(X, Y)
