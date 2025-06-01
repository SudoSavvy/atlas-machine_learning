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

    if (Transition.shape != (M, M) or Initial.shape != (M, 1)):
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
        if P == 0:
            return None, None

        # Gamma and Xi
        gamma = np.zeros((M, T))
        xi = np.zeros((M, M, T - 1))

        for t in range(T - 1):
            denom = np.sum([
                np.sum(F[i, t] * Transition[i, :] *
                       Emission[:, Observations[t + 1]] * B[:, t + 1])
                for i in range(M)
            ])
            if denom == 0:
                continue
            for i in range(M):
                numer = (F[i, t] * Transition[i, :] *
                         Emission[:, Observations[t + 1]] * B[:, t + 1])
                xi[i, :, t] = numer / denom

        gamma = np.sum(xi, axis=1)
        final_denom = np.sum(F[:, T - 1] * B[:, T - 1])
        if final_denom != 0:
            gamma = np.hstack((gamma, ((F[:, T - 1] * B[:, T - 1]) / final_denom).reshape(-1, 1)))
        else:
            gamma = np.hstack((gamma, np.zeros((M, 1))))

        # Update Initial
        Initial = gamma[:, [0]]

        # Update Transition
        for i in range(M):
            denom = np.sum(gamma[i, :-1])
            if denom == 0:
                Transition[i, :] = 0
            else:
                Transition[i, :] = np.sum(xi[i, :, :], axis=1) / denom

        # Update Emission
        for i in range(M):
            denom = np.sum(gamma[i, :])
            for k in range(N):
                mask = (Observations == k)
                numer = np.sum(gamma[i, mask])
                if denom == 0:
                    Emission[i, k] = 0
                else:
                    Emission[i, k] = numer / denom

    return Transition, Emission
