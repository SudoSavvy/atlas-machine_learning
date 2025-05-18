#!/usr/bin/env python3
import numpy as np

def pca(X, var=1.0):
    """Perform PCA on X, keeping enough components to retain variance 'var'"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    # Covariance matrix
    cov = np.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Compute cumulative variance ratio
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    # Number of components to keep
    k = np.searchsorted(cumvar, var) + 1
    # Return principal components
    return eigvecs[:, :k]
