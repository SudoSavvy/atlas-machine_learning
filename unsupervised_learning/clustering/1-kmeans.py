#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np


def initialize(X, k):
    """Documentation"""

    if not isinstance(X, np.ndarray):
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if len(X.shape) != 2:
        return None

    n, d = X.shape
    if n == 0 or d == 0:
        return None

    min_values = X.min(axis=0)
    max_values = X.max(axis=0)

    centroids = np.random.uniform(min_values, max_values, size=(k, d))

    return centroids


def kmeans(X, k, iterations=1000):
    """Documentation"""

    if k <= 0 or not isinstance(k, int):
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    n, d = X.shape

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # calculating new centroids
        new_C = np.zeros((k, d))

        for i in range(k):
            if np.any(clss == i):
                new_C[i] = X[clss == i].mean(axis=0)
            else:
                new_C[i] = np.random.uniform(
                    low=X.min(axis=0), high=X.max(axis=0), size=(d,)
                )

        if np.all(C == new_C):
            break

        C = new_C

    # return_clusters = ~np.all(new_C == 0, axis=1)
    # C = C[return_clusters]
    # clss = clss[return_clusters]

    return C, clss