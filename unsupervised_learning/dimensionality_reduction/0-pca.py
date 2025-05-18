#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """Perform PCA on X, keeping enough components to retain variance 'var'"""
    # Compute covariance matrix
    cov = np.dot(X.T, X) / (X.shape[0] - 1)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Compute cumulative variance ratio
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    # Find the smallest number of components to retain the variance
    k = np.argmax(cumvar >= var) + 1  # +1 since indexing starts at 0
    # Return the principal components
    return eigvecs[:, :k]