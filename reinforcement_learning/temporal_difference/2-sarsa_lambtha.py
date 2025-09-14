#!/usr/bin/env python3
"""SARSA(λ) Value Estimation using Eligibility Traces"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm to estimate the Q-table.

    Parameters:
    - env: The environment instance.
    - Q: A numpy.ndarray of shape (s, a) containing the Q-table.
    - lambtha: The eligibility trace factor (λ).
    - episodes: Total number of episodes to train over (default: 5000).
    - max_steps: Maximum number of steps per episode (default: 100).
    - alpha: Learning rate (default: 0.1).
    - gamma: Discount rate (default: 0.99).
    - epsilon: Initial threshold for epsilon-greedy policy (default: 1).
    - min_epsilon: Minimum value for epsilon after decay (default: 0.1).
    - epsilon_decay: Rate at which epsilon decays per episode (default: 0.05).

    Returns:
    - Q: The updated Q-table.
    """
    n_actions = Q.shape[1]

    for _ in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(Q)

        if np.random.uniform() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.random.uniform() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            eligibility *= gamma * lambtha
            eligibility[state, action] += 1

            Q += alpha * delta * eligibility

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
