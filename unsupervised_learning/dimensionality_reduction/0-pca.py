#!/usr/bin/env python3
import numpy as np

def pca(X, var=1.0):
    """Performs PCA on a dataset X to maintain the specified variance `var`"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Compute cumulative variance ratio
    cumulative_variance = np.cumsum(eigvals) / np.sum(eigvals)

    # Determine the number of components to reach desired variance
    k = np.searchsorted(cumulative_variance, var) + 1

    # Select top-k eigenvectors
    return eigvecs[:, :k]
