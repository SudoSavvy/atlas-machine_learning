#!/usr/bin/env python3
import numpy as np

def pca(X, var=1.0):
    """Performs PCA on a dataset X to maintain the specified variance `var`"""
    # Compute the covariance matrix
    cov = np.matmul(X.T, X) / (X.shape[0] - 1)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # Compute the cumulative variance
    total_variance = np.sum(eigvals_sorted)
    cumulative_variance = np.cumsum(eigvals_sorted) / total_variance

    # Determine the number of components needed to reach desired variance
    num_components = np.searchsorted(cumulative_variance, var) + 1

    # Select the corresponding eigenvectors (principal components)
    W = eigvecs_sorted[:, :num_components]

    return W
#!/usr/bin/env python3
import numpy as np

def pca(X, var=1.0):
    """Performs PCA on a dataset X to maintain the specified variance `var`"""
    # Compute the mean and center the data
    X_mean = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    cov = np.matmul(X_mean.T, X_mean) / (X.shape[0] - 1)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # Compute cumulative explained variance
    total_variance = np.sum(eigvals_sorted)
    cumulative_variance = np.cumsum(eigvals_sorted) / total_variance

    # Select number of components to maintain the desired variance
    num_components = np.searchsorted(cumulative_variance, var) + 1

    # Return the selected eigenvectors
    return eigvecs_sorted[:, :num_components]
