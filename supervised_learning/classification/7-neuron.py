#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(nx, 1)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        Z = np.dot(self.__W.T, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
        return cost

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            costs.append(cost)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                dZ = self.__A - Y
                dW = np.dot(X, dZ.T) / X.shape[1]
                db = np.sum(dZ) / X.shape[1]
                self.__W -= alpha * dW
                self.__b -= alpha * db
        if graph:
            plt.plot(np.arange(0, iterations + 1, step), costs[::step])
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.__A

    def evaluate(self, X, Y):
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.round(self.__A), cost
