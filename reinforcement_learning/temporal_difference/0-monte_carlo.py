#!/usr/bin/env python3
"""Monte Carlo Value Estimation"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.

    Parameters:
    - env: The environment instance.
    - V: A numpy.ndarray of shape (s,) containing the value estimate.
    - policy: A function that takes in a state and returns the next action.
    - episodes: Total number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.

    Returns:
    - V: The updated value estimate.
    """
    for episode in range(episodes):
        state = env.reset()
        episode_data = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        visited = set()

        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
