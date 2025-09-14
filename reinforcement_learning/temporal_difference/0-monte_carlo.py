#!/usr/bin/env python3
"""
Monte Carlo Value Estimation
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for estimating the value function.

    Args:
        env: The OpenAI Gym-like environment instance.
        V (np.ndarray): Shape (s,) array containing the current value estimates.
        policy (function): Function that takes in a state and returns an action.
        episodes (int): Total number of episodes to run.
        max_steps (int): Maximum steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.

    Returns:
        np.ndarray: Updated value estimates after Monte Carlo evaluation.
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        # Generate one full episode following the policy
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            if terminated or truncated:
                break
            state = next_state

        # Track visited states to implement first-visit MC
        visited_states = set()
        G = 0  # return
        for state, reward in reversed(episode):
            G = gamma * G + reward
            if state not in visited_states:
                visited_states.add(state)
                # Update value estimate using incremental MC update
                V[state] = V[state] + alpha * (G - V[state])

    return V
