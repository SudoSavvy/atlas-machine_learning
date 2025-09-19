#!/usr/bin/env python3
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy (action probabilities) using a weight matrix.

    Args:
        matrix (np.ndarray): shape (1, n), the state input
        weight (np.ndarray): shape (n, m), the weights of the policy network

    Returns:
        np.ndarray: shape (1, m), action probabilities after softmax
    """
    # Linear transformation
    z = np.matmul(matrix, weight)

    # Softmax for numerical stability
    z_exp = np.exp(z - np.max(z))  # subtract max for stability
    softmax = z_exp / np.sum(z_exp, axis=1, keepdims=True)

    return softmax
