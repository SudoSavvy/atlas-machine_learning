#!/usr/bin/env python3
"""Baum-Welch algorithm for Hidden Markov Models"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model.

    Args:
        Observations (np.ndarray): shape (T,) with observation indices.
        Transition (np.ndarray): shape (M, M) with transition probabilities.
        Emission (np.ndarray): shape (M, N) with emission probabilities.
        Initial (np.ndarray): shape (M, 1) with initial state probabilities.
        iterations (int): number of expectation-maximization iterations.

    Returns:
        Updated Transition, Emission matrices.
        Returns (None, None) on failure.
    """
    if (not isinstance(Observations, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Initial, np.ndarray) or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    T = Observations.shape[0]
    M, N = Emission.shape

    if (Transition.shape != (M, M) or
        Initial.shape != (M, 1)):
        return None, None

    for _ in range(iterations):
        # Forward
        F = np.zeros((M, T))
        F[:, 0] = Initial.T * Emission[:, Observations[0]]
        for t in range(1, T):
            for j in range(M):
                F[j, t] = np.sum(F[:, t - 1] * Transition[:, j]) * \
                          Emission[j, Observations[t]]

        # Backward
        B = np.zeros((M, T))
        B[:, T - 1] = 1
        for t in range(T - 2, -1, -1):
            for i in range(M):
                B[i, t] = np.sum(Transition[i, :] *
                                 Emission[:, Observations[t + 1]] *
                                 B[:, t + 1])

        # Total probability
        P = np.sum(F[:, -1])

        # Gamma and Xi
        gamma = np.zeros((M, T))
        xi = np.zeros((M, M, T - 1))

        for t in range(T - 1):
            denom = np.sum(F[:, t] * B[:, t])
            for i in range(M):
                gamma[i, t] = (F[i, t] * B[i, t]) / denom
                xi[i, :, t] = (F[i, t] *
                               Transition[i, :] *
                               Emission[:, Observations[t + 1]] *
                               B[:, t + 1]) / np.sum(
                                   F[:, t] * Transition * Emission[:, Observations[t + 1]] * B[:, t + 1].T)

        denom = np.sum(F[:, T - 1] * B[:, T - 1])
        for i in range(M):
            gamma[i, T - 1] = (F[i, T - 1] * B[i, T - 1]) / denom

        # Re-estimate Initial
        Initial = gamma[:, [0]]

        # Re-estimate Transition
        for i in range(M):
            for j in range(M):
                numer = np.sum(xi[i, j, :])
                denom = np.sum(gamma[i, :-1])
                Transition[i, j] = numer / denom

        # Re-estimate Emission
        for i in range(M):
            for k in range(N):
                mask = (Observations == k)
                Emission[i, k] = np.sum(gamma[i, mask]) / np.sum(gamma[i, :])

    return Transition, Emission
