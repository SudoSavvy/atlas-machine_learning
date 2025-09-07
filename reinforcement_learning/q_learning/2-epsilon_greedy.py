#!/usr/bin/env python3
"""
Module for epsilon-greedy action selection.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects the next action using the epsilon-greedy policy.

    Args:
        Q (numpy.ndarray): Q-table of shape (n_states, n_actions).
        state (int): Current state.
        epsilon (float): Probability of exploring.

    Returns:
        int: The chosen action index.
    """
    p = np.random.uniform(0, 1)
    n_actions = Q.shape[1]

    if p < epsilon:
        # Explore: pick a random action
        action = np.random.randint(0, n_actions)
    else:
        # Exploit: pick the best known action
        action = np.argmax(Q[state])
    return action
