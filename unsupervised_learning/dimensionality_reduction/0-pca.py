#!/usr/bin/env python3
"""
Performs PCA on dataset X to maintain a given fraction of variance.
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X.

    Parameters:
    - X: numpy.ndarray of shape (n, d), zero-mean data
    - var: float, fraction of variance to maintain

    Returns:
    - W: numpy.ndarray of shape (d, nd), weights matrix
    """

    # Covariance matrix
    covariance_matrix = np.dot(X.T, X) / (X.shape[0] - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Cumulative variance ratio
    cum_var_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Number of components to keep at least 'var' variance
    nd = np.argmax(cum_var_ratio >= var) + 1

    # Select top components
    W = eigenvectors[:, :nd]

    return W
