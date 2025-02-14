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
    mean = np.mean(x, axis=0)
    std_dev = np.std(x, axis=0, ddof=0)
    return (mean, std_dev)


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix.

    Parameters:
    X (numpy.ndarray): A matrix of shape (d, nx) to normalize
        d is the number of data points
        nx is the number of features
    m (numpy.ndarray): A numpy array of shape (nx,) containing
    the mean of all features of X
    s (numpy.ndarray): A numpy array of shape (nx,) containing
    the standard deviation of all features of X

    Returns:
    numpy.ndarray: The normalized X matrix
    """
    return (X - m) / s


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Parameters:
    X (numpy.ndarray): The first matrix of shape (m, nx) to shuffle
        m is the number of data points
        nx is the number of features in X
    Y (numpy.ndarray): The second matrix of shape (m, ny) to shuffle
        m is the same number of data points as in X
        ny is the number of features in Y

    Returns:
    tuple: The shuffled X and Y matrices
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
