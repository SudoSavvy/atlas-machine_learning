#!/usr/bin/env python3
"""Determines the steady state probabilities of a regular Markov chain"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.

    Args:
        P (numpy.ndarray): 2D square array of shape (n, n) representing
                           the transition matrix.

    Returns:
        numpy.ndarray: Array of shape (1, n) containing the steady state
                       probabilities, or None on failure.
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    n, m = P.shape
    if n != m:
        return None

    # Check if P is a valid stochastic matrix
    if not np.allclose(P.sum(axis=1), 1):
        return None

    # Check if it's regular: some power of P has all entries > 0
    power = np.linalg.matrix_power(P, 100)
    if not np.all(power > 0):
        return None

    # Solve for steady state vector π such that πP = π and sum(π) = 1
    # Transpose to get P^T, so we solve (P^T - I)^T π = 0 with constraint sum(π) = 1
    A = P.T - np.eye(n)
    A = np.vstack([A, np.ones((1, n))])
    b = np.zeros((n + 1,))
    b[-1] = 1

    try:
        steady = np.linalg.lstsq(A, b, rcond=None)[0]
        return steady.reshape(1, n)
    except Exception:
        return None
