#!/usr/bin/env python3
"""TD(λ) Value Estimation using Eligibility Traces"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm to estimate the value function.

    Parameters:
    - env: The environment instance.
    - V: A numpy.ndarray of shape (s,) containing the value estimate.
    - policy: A function that takes in a state and returns the next
      action to take.
    - lambtha: The eligibility trace factor (λ).
    - episodes: Total number of episodes to train over (default: 5000).
    - max_steps: Maximum number of steps per episode (default: 100).
    - alpha: Learning rate (default: 0.1).
    - gamma: Discount rate (default: 0.99).

    Returns:
    - V: The updated value estimate.
    """
    for _ in range(episodes):
        state = env.reset()[0]  # Handles (obs, info) format
        eligibility = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace
            eligibility *= gamma * lambtha
            eligibility[state] += 1

            # Update value function
            V += alpha * delta * eligibility

            if terminated or truncated:
                break
            state = next_state

    return V
