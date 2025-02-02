#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int) or nx < 1:
            raise TypeError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) < 1 or not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")
        
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        for l in range(self.L):
            key_W = f"W{l + 1}"
            key_b = f"b{l + 1}"
            
            self.weights[key_W] = np.random.randn(layers[l], nx if l == 0 else layers[l - 1]) * np.sqrt(1 / (nx if l == 0 else layers[l - 1]))
            self.weights[key_b] = np.zeros((layers[l], 1))
    
    def forward_prop(self, X):
        self.cache["A0"] = X
        
        for l in range(1, self.L + 1):
            W, b = self.weights[f"W{l}"], self.weights[f"b{l}"]
            A_prev = self.cache[f"A{l - 1}"]
            Z = np.matmul(W, A_prev) + b
            
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True) if l == self.L else 1 / (1 + np.exp(-Z))
            self.cache[f"A{l}"] = A
        
        return A, self.cache
    
    def cost(self, Y, A):
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m  # Cross-entropy loss
    
    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        return np.argmax(A, axis=0), self.cost(Y, A)
    
    def gradient_descent(self, Y, alpha=0.05):
        m = Y.shape[1]
        dZ = self.cache[f"A{self.L}"] - Y
        
        for l in reversed(range(1, self.L + 1)):
            A_prev = self.cache[f"A{l - 1}"]
            W = self.weights[f"W{l}"]
            dW, db = np.matmul(dZ, A_prev.T) / m, np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                dZ = np.matmul(W.T, dZ) * (A_prev * (1 - A_prev))
            
            self.weights[f"W{l}"] -= alpha * dW
            self.weights[f"b{l}"] -= alpha * db
    
    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True):
        if not isinstance(iterations, int) or iterations <= 0:
            raise TypeError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise TypeError("alpha must be a positive float")
        
        costs = []
        
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, alpha)
            
            if verbose and i % 100 == 0:
                cost = self.cost(Y, self.cache[f"A{self.L}"])
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