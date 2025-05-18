#!/usr/bin/env python3
"""Module for calculating mean and covariance of a dataset."""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    - X (numpy.ndarray): shape (n, d), dataset with n data points and d dimensions.

    Returns:
    - mean (numpy.ndarray): shape (1, d), the mean of the data set
    - cov (numpy.ndarray): shape (d, d), the covariance matrix of the data set

    Raises:
    - TypeError: if X is not a 2D numpy.ndarray
    - ValueError: if X contains fewer than 2 data points
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
