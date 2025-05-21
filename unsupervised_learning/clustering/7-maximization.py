#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np


def maximization(X, g):
    """Documentation"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    return None, None, None