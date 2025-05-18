#!/usr/bin/env python3
import numpy as np

def pca(X, var=1.0):
    """Performs PCA on dataset X to retain specified variance `var`"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = np.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Cumulative variance ratio
    cumulative_variance = np.cumsum(eigvals) / np.sum(eigvals)

    # Number of components needed to retain desired variance
    k = np.searchsorted(cumulative_variance, var) + 1

    # Return principal components
    return eigvecs[:, :k]
