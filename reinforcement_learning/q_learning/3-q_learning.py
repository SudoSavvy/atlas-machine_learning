#!/usr/bin/env python3
"""
Module for training a Q-learning agent on FrozenLake.
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Trains a Q-learning agent on the given FrozenLake environment.

    Args:
        env (gym.Env): The FrozenLake environment.
        Q (numpy.ndarray): Q-table to update, shape (n_states, n_actions).
        episodes (int): Number of episodes to train.
        max_steps (int): Max steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial exploration rate.
        min_epsilon (float): Minimum epsilon allowed.
        epsilon_decay (float): Rate of exponential epsilon decay.

    Returns:
        tuple:
            - Q (numpy.ndarray): The updated Q-table.
            - total_rewards (list[float]): Rewards collected per episode.
    """
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            # Choose action using epsilon-greedy
            action = epsilon_greedy(Q, state, epsilon)

            # Take step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Custom rule: falling into a hole gives -1 reward
            if reward == 0 and terminated:
                reward = -1

            # Update Q-value (Q-learning update rule)
            best_next_action = np.max(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * best_next_action - Q[state, action]
            )

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q, total_rewards
