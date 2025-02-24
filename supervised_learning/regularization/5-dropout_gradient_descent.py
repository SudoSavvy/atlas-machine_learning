#!/usr/bin/env python3
"""Gradient Descent with Dropout Module"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network with Dropout
    regularization using gradient descent:

    Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
    classes is the number of classes
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs and dropout masks of each layer of
    the neural network
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    -> All layers use the tanh activation function except the last, which uses
    the softmax activation function
    -> The weights of the network should be updated in place

    """
    # calculate m as the inverse of the number of samples
    m = 1 / Y.shape[1]
    # compute dZ using values of cache
    dZ = cache[f'A{L}'] - Y

    # iterate over the layers in reverse backprop
    for layer in reversed(range(1, L + 1)):
        A_prev = cache[f'A{layer - 1}']
        w, b = f'W{layer}', f'b{layer}'
        dW = m * np.matmul(dZ, A_prev.T)
        db = m * np.sum(dZ, axis=1, keepdims=True)

        # if not the first layer, compute the gradient for prev layer
        if layer > 1:
            # backprop through linear component
            dZ = np.matmul(weights[w].T, dZ)
            # backprop using tanh
            dZ *= (1 - np.power(A_prev, 2))
            # apply dropout mask then scale
            dZ *= (cache[f'D{layer - 1}'] / keep_prob)

        # updates the weights and biases using gradient descent
        weights[w] -= alpha * dW
        weights[b] -= alpha * db
        