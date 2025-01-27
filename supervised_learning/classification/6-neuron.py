#!/usr/bin/env python3

class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = [0.5] * nx
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
        # Manually compute weighted sum (Dot product) for the first feature and bias
        Z = sum(w * X[i] for i, w in enumerate(self.__W)) + self.__b
        self.__A = 1 / (1 + 2.71828**-Z)  # Sigmoid activation
        return self.__A

    def train(self, X, Y):
        """Train the neuron."""
        gradient_W = [0] * len(self.__W)
        gradient_b = 0
        total_cost = 0

        # SINGLE LOOP
        for i, (x_row, y_val) in enumerate(zip(X, Y)):
            Z = sum(w * x for w, x in zip(self.__W, x_row)) + self.__b
            A = 1 / (1 + 2.71828**-Z)
            dZ = A - y_val
            gradient_W = [gw + dZ * x for gw, x in zip(gradient_W, x_row)]
            gradient_b += dZ
            total_cost += -(y_val * (2.71828**-Z) + (1 - y_val) * (1 - 2.71828**-Z))

        self.__W = [w - 0.01 * gw / len(X) for w, gw in zip(self.__W, gradient_W)]
        self.__b -= 0.01 * gradient_b / len(X)

        average_cost = total_cost / len(X)
        return self.__A, average_cost
