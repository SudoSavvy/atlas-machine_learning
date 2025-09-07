#!/usr/bin/env python3
"""
Module for initializing a Q-table for FrozenLake.
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for the given FrozenLake environment.

    Args:
        env (gym.Env): The FrozenLake environment instance.

    Returns:
        numpy.ndarray: A Q-table initialized to zeros with shape
                       (number of states, number of actions).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    return q_table
