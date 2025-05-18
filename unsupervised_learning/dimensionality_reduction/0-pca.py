#!/usr/bin/env python3
import numpy as np

def pca(X, var=1.0):
    """Performs PCA on dataset X to retain the specified variance `var`"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = np.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Cumulative explained variance
    cumulative_variance = np.cumsum(eigvals) / np.sum(eigvals)

    # Components to retain the desired variance
    k = np.searchsorted(cumulative_variance, var) + 1

    # Return the first k principal components
    return eigvecs[:, :k]
