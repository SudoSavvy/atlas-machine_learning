#!/usr/bin/env python3
"""Viterbi algorithm for Hidden Markov Models"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for an HMM.

    Args:
        Observation (np.ndarray): shape (T,) containing index of observations.
            T is the number of observations.
        Emission (np.ndarray): shape (N, M) with emission probabilities.
            Emission[i, j] is the prob. of observing j from state i.
        Transition (np.ndarray): shape (N, N) with transition probabilities.
            Transition[i, j] is the prob. of transitioning from state i to j.
        Initial (np.ndarray): shape (N, 1) with initial state probabilities.

    Returns:
        path (list): most likely sequence of hidden states.
        P (float): probability of obtaining the path sequence.
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

    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)

    V[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for j in range(N):
            prob = V[:, t - 1] * Transition[:, j]
            B[j, t] = np.argmax(prob)
            V[j, t] = np.max(prob) * Emission[j, Observation[t]]

    P = np.max(V[:, -1])
    last_state = np.argmax(V[:, -1])

    path = [last_state]
    for t in range(T - 1, 0, -1):
        last_state = B[last_state, t]
        path.insert(0, last_state)

    return path, P
