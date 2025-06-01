#!/usr/bin/env python3
"""Baum-Welch algorithm for HMM"""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model"""
    try:
        T = Observations.shape[0]
        N, M = Emission.shape

        for _ in range(iterations):
            # Forward algorithm
            F = np.zeros((N, T))
            F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
            for t in range(1, T):
                for j in range(N):
                    F[j, t] = np.dot(F[:, t - 1], Transition[:, j]) * Emission[j, Observations[t]]

            # Backward algorithm
            B = np.zeros((N, T))
            B[:, T - 1] = 1
            for t in reversed(range(T - 1)):
                for i in range(N):
                    B[i, t] = np.sum(Transition[i, :] *
                                     Emission[:, Observations[t + 1]] * B[:, t + 1])

            # Compute xi and gamma
            xi = np.zeros((N, N, T - 1))
            for t in range(T - 1):
                denominator = np.dot(np.dot(F[:, t], Transition) *
                                     Emission[:, Observations[t + 1]], B[:, t + 1])
                if denominator == 0:
                    continue
                for i in range(N):
                    numerator = F[i, t] * Transition[i, :] * \
                        Emission[:, Observations[t + 1]] * B[:, t + 1]
                    xi[i, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            # Add last gamma for time T - 1
            prod = F[:, T - 1] * B[:, T - 1]
            gamma = np.hstack((gamma, prod[:, None] / np.sum(prod)))

            # Update model parameters
            Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1, keepdims=True)

            Emission = np.zeros((N, M))
            for t in range(T):
                Emission[:, Observations[t]] += gamma[:, t]
            Emission /= np.sum(gamma, axis=1, keepdims=True)

        return Transition, Emission
    except Exception:
        return None, None
