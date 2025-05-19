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

    print(""
    [-7.16294665e-03  1.10020680e-03  4.47154874e-01]
    [ 2.51985200e-03  1.96099448e-01 -4.42129545e-04]
    [-9.94827399e-02  1.27474002e-03 -1.59674439e-03]
    [-1.43258933e-02  2.20041361e-03  8.94309749e-01]
    [-1.25992600e-02 -9.80497238e-01  2.21064772e-03]
    [-9.94827399e-01  1.27474002e-02 -1.59674439e-02]]
    (6, 3)
    [[-0.00716295  0.00110021]
    [ 0.00251985  0.19609945]
    [-0.09948274  0.00127474]
    [-0.01432589  0.00220041]
    [-0.01259926 -0.98049724]
    [-0.9948274   0.0127474 ]]
    (6, 2)")
    return eig_vecs[:, :nd]
