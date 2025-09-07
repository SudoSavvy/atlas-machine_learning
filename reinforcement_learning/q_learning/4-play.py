#!/usr/bin/env python3
"""
Module for playing an episode with a trained Q-learning agent.
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Runs one episode with the trained agent exploiting the Q-table.

    Args:
        env (gym.Env): The FrozenLake environment (with render_mode="ansi").
        Q (numpy.ndarray): The trained Q-table.
        max_steps (int): Maximum number of steps in the episode.

    Returns:
        tuple:
            - total_reward (float): The total reward collected in the episode.
            - frames (list[str]): List of rendered environment states.
    """
    state, _ = env.reset()
    frames = [env.render()]
    total_reward = 0

    for _ in range(max_steps):
        # Always exploit (no epsilon-greedy here)
        action = np.argmax(Q[state])

        # Take step
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Track reward and render state
        total_reward += reward
        frames.append(env.render())

        state = next_state

        if terminated or truncated:
            break

    return total_reward, frames
