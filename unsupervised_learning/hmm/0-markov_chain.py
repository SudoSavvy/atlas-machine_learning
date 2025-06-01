#!/usr/bin/env python3
"""Markov Chain Probability Calculation"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular
    state after a specified number of iterations.

    Args:
        P (numpy.ndarray): 2D square array of shape (n, n) representing
                           the transition matrix.
        s (numpy.ndarray): 1D array of shape (1, n) representing the initial
                           state probabilities.
        t (int): Number of iterations (steps) to compute.

    Returns:
        numpy.ndarray: Array of shape (1, n) with the resulting state
                       probabilities after t iterations, or None on failure.
    """
    if not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray):
        return None

    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]

    if s.shape != (1, n):
        return None

    if not isinstance(t, int) or t < 0:
        return None

    try:
        result = s.copy()
        for _ in range(t):
            result = np.matmul(result, P)
        return result
    except Exception:
        return None
