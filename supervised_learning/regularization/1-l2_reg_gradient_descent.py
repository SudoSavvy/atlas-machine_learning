#!/usr/bin/env python3
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (float): Cost of the network without L2 regularization.
    lambtha (float): Regularization parameter.
    weights (dict): Dictionary of weights and biases of the neural network.
    L (int): Number of layers in the neural network.
    m (int): Number of data points used.

    Returns:
    float: Cost of the network accounting for L2 regularization.
    """
    l2_norm = sum(np.linalg.norm(weights[f"W{i}"])**2 for i in range(1, L + 1))
    l2_cost = cost + (lambtha / (2 * m)) * l2_norm

    return l2_cost

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Parameters:
    Y (numpy.ndarray): One-hot array of shape (classes, m) containing
    correct labels.
    weights (dict): Dictionary of the weights and biases of the neural
    network.
    cache (dict): Dictionary of the outputs of each layer.
    alpha (float): Learning rate.
    lambtha (float): L2 regularization parameter.
    L (int): Number of layers in the network.

    Updates weights and biases in place.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"] if i > 1 else cache["A0"]
        dW = (np.dot(dZ, A_prev.T) + lambtha * weights[f"W{i}"]) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        if i > 1:
            dZ = np.dot(weights[f"W{i}"].T, dZ) * (1 - cache[f"A{i-1}"]**2)
