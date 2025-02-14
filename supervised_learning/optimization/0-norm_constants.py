#!/usr/bin/env python3
"""
Function to calculate the normalization constants of a matrix.
"""
import numpy as np


def normalization_constants(x):
    """
    Calculates the mean and standard deviation of each feature in a matrix.

    Parameters:
    X (numpy.ndarray): A matrix of shape (m, nx) where m is the number of
                        data points and nx is the number of features.

    Returns:
    tuple: A tuple containing the mean and standard deviation of each feature,
           respectively.
    """
    mean =np.mean(x,axis=0)
    std_dev = np.std(x, axis=0)
    return mean,