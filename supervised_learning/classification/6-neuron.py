#!/usr/bin/env python3
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = [[0.5] * nx]
        self.__b = 0
        self.__A = 0

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
        """Perform forward propagation."""
        Z = sum([self.__W] * len(X))  # Replace with appropriate computation
        self.__A = 1 / (1 + pow(2.71828, -Z))  # Correct activation computation

    def train(self, X, Y):
        """Train the neuron."""
        # Dummy implementation; replace with actual training logic
        for _ in range(100):  # Example iteration loop
            self.forward_prop(X)
            self.__W = [[w - 0.01 for w in weights] for weights in self.__W]  # Example weight update
        return self.__A, 0.5  # Return example output values
