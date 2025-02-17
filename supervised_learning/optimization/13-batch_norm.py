#!/usr/bin/env python3

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch
    normalization.

    Args:
        Z (numpy.ndarray): Shape (m, n) to be normalized.
            m: Number of data points.
            n: Number of features in Z.
        gamma (numpy.ndarray): Shape (1, n), scales used for batch
        normalization.
        beta (numpy.ndarray): Shape (1, n), offsets used for batch
        normalization.
        epsilon (float): Small number to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix.
    """
    # Calculate mean and variance along the batch (axis=0)
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # Normalize Z
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    Z_batch_norm = gamma * Z_normalized + beta

    return Z_batch_norm
