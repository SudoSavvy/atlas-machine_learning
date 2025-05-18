#!/usr/bin/env python3
"""Module to calculate likelihood of severe side effects using binomial distribution."""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of observing x patients with side effects
    out of n total patients given hypothetical probabilities in P.

    Parameters:
    - x (int): number of patients with severe side effects
    - n (int): total number of patients observed
    - P (numpy.ndarray): 1D array of hypothetical probabilities

    Returns:
    - numpy.ndarray: likelihoods for each probability in P

    Raises:
    - TypeError: if P is not a 1D numpy.ndarray
    - ValueError: if n is not a positive integer
    - ValueError: if x is not an integer >= 0
    - ValueError: if x > n
    - ValueError: if any values in P are not in [0, 1]
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient: C(n, x) = n! / (x!(n - x)!)
    coeff = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n - x))

    # Likelihood: C(n, x) * p^x * (1-p)^(n-x)
    likelihoods = coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
