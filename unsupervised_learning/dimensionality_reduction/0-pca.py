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
    cov = np.cov(X, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, sorted_idx]

    # Cumulative variance ratio
    total = np.sum(eig_vals)
    cum_var = np.cumsum(eig_vals) / total

    # Find number of components to keep at least var fraction
    nd = np.argmax(cum_var >= var) + 1

    return eig_vecs[:, :nd]
