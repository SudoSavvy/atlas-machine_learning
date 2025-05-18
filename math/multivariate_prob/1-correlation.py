#!/usr/bin/env python3
"""Module for calculating the correlation matrix from a covariance matrix."""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Parameters:
    - C (numpy.ndarray): shape (d, d), the covariance matrix

    Returns:
    - numpy.ndarray: shape (d, d), the correlation matrix

    Raises:
    - TypeError: if C is not a numpy.ndarray
    - ValueError: if C is not a 2D square matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_devs = np.sqrt(np.diag(C))
    denom = np.outer(std_devs, std_devs)

    # Avoid division by zero by replacing zeros in denom with np.nan
    # Then replacing any resulting nan in correlation matrix with 0
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = C / denom
        corr[np.isnan(corr)] = 0

    return corr
