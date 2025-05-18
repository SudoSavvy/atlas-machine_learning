#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """Performs PCA on a dataset
    
    Args:
        X (np.ndarray): shape (n, d), dataset with zero mean
        var (float): fraction of the variance that PCA should maintain
    
    Returns:
        np.ndarray: shape (d, nd), weights matrix that maintains var fraction of variance
    """
    # Compute the covariance matrix
    cov = np.cov(X, rowvar=False)

    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_indices]
    eig_vecs = eig_vecs[:, sorted_indices]

    # Compute cumulative variance ratio
    total = np.sum(eig_vals)
    cum_var = np.cumsum(eig_vals) / total

    # Find the number of dimensions to retain
    nd = np.searchsorted(cum_var, var) + 1

    # Return the first nd eigenvectors (principal components)
    return eig_vecs[:, :nd]
