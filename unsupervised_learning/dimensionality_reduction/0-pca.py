#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """Perform PCA on X to retain specified variance."""
    # Compute covariance matrix (X is already centered)
    cov = np.dot(X.T, X) / (X.shape[0] - 1)
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Calculate cumulative explained variance
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    # Find the smallest number of components needed
    k = np.searchsorted(cumvar, var) + 1
    # Handle case where var is 1.0 to include all components
    if var >= 1.0:
        k = len(eigvals)
    return eigvecs[:, :k]