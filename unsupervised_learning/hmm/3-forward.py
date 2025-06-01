#!/usr/bin/env python3
"""Performs the forward algorithm for a hidden Markov model"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Args:
        Observation (numpy.ndarray): shape (T,) with indices of observations.
            T is the number of observations.
        Emission (numpy.ndarray): shape (N, M) with emission probabilities.
            Emission[i, j] is the prob. of observing j from hidden state i.
        Transition (numpy.ndarray): shape (N, N) with transition probabilities.
            Transition[i, j] is the prob. of transitioning from i to j.
        Initial (numpy.ndarray): shape (N, 1) with starting state probabilities.

    Returns:
        P (float): likelihood of the observations given the model.
        F (numpy.ndarray): shape (N, T) with forward path probabilities.
            F[i, j] is the probability of being in state i at time j.
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

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.dot(F[:, t - 1], Transition[:, j]) * \
                      Emission[j, Observation[t]]

    P = np.sum(F[:, -1])

    return P, F
