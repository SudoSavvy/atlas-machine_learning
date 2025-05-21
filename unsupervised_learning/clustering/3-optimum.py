#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Documentation"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is not None and (not isinstance(kmax, int) or kmax <= kmin):
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if kmax <= kmin:
        return None, None

    return None, None