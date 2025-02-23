#!/usr/bin/env python3
"""Here's some documentation
Y - one-hot numpy.ndarray of shape (classes, m)
    contains the correct labels for the data
classes - number of classes
m - number of data points
weights - dictionary of weights and biases of the neural network
cache - dictionary of outputs of each layer of the neural network
alpha - learning rate
lambtha - L2 regularization parameter
L - number of layers of the network"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """And some more documentation"""

    m = Y.shape[1]

    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        W = weights['W' + str(i)]

        dW = m ** -1 * np.dot(dZ, A_prev.T) + lambtha / m * W
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        dZ = dA_prev * (1 - np.power(A_prev, 2))
        