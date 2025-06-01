#!/usr/bin/env python3
"""Determines if a Markov chain is absorbing"""

import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing.

    Args:
        P (numpy.ndarray): 2D square array of shape (n, n) representing
                           the transition matrix.

    Returns:
        bool: True if the chain is absorbing, False otherwise or on failure.
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False

    n, m = P.shape
    if n != m:
        return False

    # Check for valid transition matrix: each row should sum to 1
    if not np.allclose(P.sum(axis=1), 1):
        return False

    # Identify absorbing states: where P[i][i] == 1 and the rest of row i is 0
    absorbing_states = np.isclose(np.diag(P), 1) & np.allclose(P - np.diag(np.diag(P)), 0, atol=1e-8)

    if not np.any(absorbing_states):
        return False

    # Try to determine if from every non-absorbing state we can reach an absorbing state
    reachable = np.copy(P)
    for _ in range(n):
        reachable = np.matmul(reachable, P)

    for i in range(n):
        if not absorbing_states[i]:
            # Check if any absorbing state is reachable from state i
            if not np.any(reachable[i] * absorbing_states):
                return False

    return True
