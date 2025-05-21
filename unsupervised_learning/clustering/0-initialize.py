#!/usr/bin/env python3
"""
Initializes cluster centroids for K-means using a
uniform distribution.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
    - X: np.ndarray of shape (n, d), the input dataset
    - k: positive int, the number of clusters

    Returns:
    - np.ndarray of shape (k, d) with the initialized centroids,
      or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    centroids = np.random.uniform(low=min_vals, high=max_vals, size=(k, X.shape[1]))
    return centroids
