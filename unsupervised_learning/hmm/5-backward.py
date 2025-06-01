#!/usr/bin/env python3
"""Backward algorithm for Hidden Markov Models"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model.

    Args:
        Observation (np.ndarray): shape (T,) with indices of observations.
            T is the number of observations.
        Emission (np.ndarray): shape (N, M) with emission probabilities.
            Emission[i, j] is prob. of observing j from hidden state i.
        Transition (np.ndarray): shape (N, N) with transition probabilities.
            Transition[i, j] is prob. of transitioning from i to j.
        Initial (np.ndarray): shape (N, 1) with initial state probabilities.

    Returns:
        P (float): likelihood of the observations given the model.
        B (np.ndarray): shape (N, T) with backward path probabilities.
            B[i, j] is the prob. of the future observations from state i at t=j.
        Returns (None, None) on failure.
    """
    if (not isinstance(Observation, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
        not isinstance(Initial, np.ndarray)):
        return None, None

    if len(Observation.shape) != 1 or \
       len(Emission.shape) != 2 or \
       len(Transition.shape) != 2 or \
       len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None

    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(B[:, t + 1] *
                             Transition[i, :] *
                             Emission[:, Observation[t + 1]])

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
