#!/usr/bin/env python3
"""Performs PCA on zero-mean dataset X to maintain specified variance fraction."""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on dataset X.

    Parameters:
    - X (numpy.ndarray): shape (n, d), data with zero mean across dimensions
    - var (float): fraction of variance to maintain

    Returns:
    - W (numpy.ndarray): shape (d, nd), weights matrix that maintains 'var' fraction of variance
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if not (0 < var <= 1):
        raise ValueError("var must be a float between 0 and 1")

    n, d = X.shape
    # Covariance matrix (d x d)
    cov = (X.T @ X) / (n - 1)

    # Eigen decomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute cumulative variance ratio
    cum_var_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    # Find number of components to maintain desired variance
    nd = np.searchsorted(cum_var_ratio, var) + 1

    # Select the first nd eigenvectors
    W = eigenvectors[:, :nd]

    return W
