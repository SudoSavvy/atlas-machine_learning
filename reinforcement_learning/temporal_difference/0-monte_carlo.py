#!/usr/bin/env python3
"""Monte Carlo Value Estimation using First-Visit Method.

This script implements the Monte Carlo algorithm to estimate the value
function V(s) for a given environment and policy. It uses first-visit
Monte Carlo with incremental updates based on observed returns.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Estimates the value function V(s) using the Monte Carlo method.

    Parameters:
    - env: The environment instance following the OpenAI Gym interface.
    - V: A numpy.ndarray of shape (s,) containing the initial value estimates.
    - policy: A function that takes a state and returns an action.
    - episodes: Total number of episodes to run (default: 5000).
    - max_steps: Maximum number of steps per episode (default: 100).
    - alpha: Learning rate for updating value estimates (default: 0.1).
    - gamma: Discount factor for future rewards (default: 0.99).

    Returns:
    - V: The updated numpy.ndarray containing value estimates for each state.
    """
    for i in range(episodes):
        episode = []
        state = env.reset()[0]  # Handles environments that return (obs, info)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, reward))

            if terminated or truncated:
                break
            state = next_state

        G = 0
        episode = np.array(episode, dtype=int)

        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state not in episode[:i, 0]:
                V[state] += alpha * (G - V[state])

    return V
