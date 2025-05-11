#!/usr/bin/env python3
"""
This module defines a function to evaluate the definiteness
of a square symmetric matrix using eigenvalues.
"""

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a symmetric matrix.

    Args:
        matrix (numpy.ndarray): Matrix to classify.

    Returns:
        str or None: One of the following classifications:
            - "Positive definite"
            - "Positive semi-definite"
            - "Negative semi-definite"
            - "Negative definite"
            - "Indefinite"
            or None if matrix is not valid.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
