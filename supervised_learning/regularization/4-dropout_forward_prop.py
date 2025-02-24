#!/usr/bin/env python3
"""Forward Propagation with Dropout Module"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward propagation using Dropout:

    X = a numpy.ndarray of shape (nx, m) containing the input data
    for the network
    nx = the number of input features
    m = the number of data points
    weights = a dictionary of the weights and biases of the neural network
    L = the number of layers in the network
    keep_prob = the probability that a node will be kept
    -> All layers except the last should use the tanh activation function
    -> The last layer should use the softmax activation function

    """
    # init the cache with X
    cache = {'A0': X}

    # iterate through each layer
    for layer in range(1, L + 1):
        # Compute Z
        Z = np.matmul(weights['W' + str(layer)],
                      cache['A' + str(layer - 1)]) + weights['b' + str(layer)]

        # If it's the last layer, then use the softmax activation function
        if layer == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A' + str(layer)] = exp_Z / np.sum(exp_Z,
                                                     axis=0,
                                                     keepdims=True)
        else:
            # If not then use the tanh activation function
            A = np.tanh(Z)

            # Add the dropout mask and use it in the output, then scale
            D = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A = (A * D) / keep_prob

            # Save the dropout mask, activation to the cache
            cache['A' + str(layer)] = A
            cache['D' + str(layer)] = D

    # returns the cache containing activations and dropout mask
    return (cache)