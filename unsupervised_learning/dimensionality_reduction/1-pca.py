#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np


def pca(X, ndim):
    """And some more documentation"""
    cov_matrix = np.cov(X, rowvar=False)

    eigenvals, eigenvects = np.linalg.eigh(cov_matrix)

    sorted_idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sorted_idx]
    eigenvects = eigenvects[:, sorted_idx]

    W = eigenvects[:, :ndim]

    T = np.dot(X, W)

    return T