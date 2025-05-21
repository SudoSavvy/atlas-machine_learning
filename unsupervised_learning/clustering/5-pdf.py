#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np


def pdf(X, m, S):
    """Documentation"""

    if not isinstance(X, np.ndarray):
        return None
    if not isinstance(m, np.ndarray):
        return None
    if not isinstance(S, np.ndarray):
        return None

    if X.ndim != 2 or m.ndim != 1 or S.ndim != 2:
        return None

    if X.shape[1] != m.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    if S.shape[0] != m.shape[0]:
        return None

    n, d = X.shape

    det_S = np.linalg.det(S)
    if det_S == 0:
        return None

    inv_S = np.linalg.inv(S)

    norm_constant = 1 / np.sqrt((2 * np.pi) ** d * det_S)

    diff = X - m

    Mah_distance = np.sum(diff @ inv_S * diff, axis=1)

    P = norm_constant * np.exp(-0.5 * Mah_distance)

    P = np.maximum(P, 1e-300)

    return P