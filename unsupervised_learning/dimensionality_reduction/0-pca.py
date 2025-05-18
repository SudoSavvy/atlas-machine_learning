#!/usr/bin/env python3
import numpy as np

def pca(X, var=0.95):
    """Performs PCA on a dataset and returns the weights matrix maintaining specified variance."""
    # Perform SVD on the zero-mean data
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Calculate the explained variance ratios
    explained_var_ratios = (S ** 2) / np.sum(S ** 2)
    
    # Compute cumulative explained variance
    cumulative_var = np.cumsum(explained_var_ratios)
    
    # Find the smallest number of components that explain at least 'var' variance
    if np.max(cumulative_var) < var:
        k = len(cumulative_var)
    else:
        k = np.argmax(cumulative_var >= var) + 1
    
    # Extract the top k components (rows of Vt) and transpose to get the weights matrix
    W = Vt[:k, :].T
    
    return W