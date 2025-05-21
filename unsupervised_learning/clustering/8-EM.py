#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Documentation"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    return None, None, None, None, None