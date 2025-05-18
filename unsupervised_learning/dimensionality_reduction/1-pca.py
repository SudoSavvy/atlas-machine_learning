#!/usr/bin/env python3
"""
Performs PCA on a dataset to reduce its dimensionality.
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Parameters:
    - X: numpy.ndarray of shape (n, d)
        The dataset where n is the number of data points
        and d is the number of dimensions in each point.
    - ndim: int
        The new dimensionality of the transformed X.

    Returns:
    - T: numpy.ndarray of shape (n, ndim)
        The transformed version of X.
    """
    # Compute the mean of X and center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Compute the covariance matrix
    covariance_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    # Select top 'ndim' eigenvectors
    W = eigenvectors_sorted[:, :ndim]

    # Project data onto principal components
    T = np.dot(X_centered, W)

    return T
