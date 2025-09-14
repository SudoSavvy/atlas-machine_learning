#!/usr/bin/env python3
"""
Monte Carlo Value Estimation
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs Monte Carlo value estimation using first-visit MC
    and true averaging of returns.

    Args:
        env: Environment instance with reset() and step() methods.
        V (np.ndarray): Value function estimate (shape (s,)).
        policy (function): Function that takes a state and returns an action.
        episodes (int): Number of episodes to run.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): (Ignored for true averaging)
        gamma (float): Discount factor.

    Returns:
        np.ndarray: Updated value estimates.
    """
    # Track returns for each state to compute mean
    returns_sum = np.zeros_like(V, dtype=np.float64)
    returns_count = np.zeros_like(V, dtype=np.int32)

    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        # Generate one full episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Calculate returns G and update first-visit states
        visited = set()
        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]

    return V
