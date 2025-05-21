#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np


def variance(X, C):
    """Documentation"""
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if X.ndim != 2 or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2

    min_dis = np.min(distances, axis=1)

    var = np.sum(min_dis)

    return var