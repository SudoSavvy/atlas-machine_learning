#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Documentation"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    return None, None, None