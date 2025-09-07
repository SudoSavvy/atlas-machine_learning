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
            - total_reward (float): Total reward collected in the episode.
            - frames (list[str]): Rendered environment states.
    """
    state, _ = env.reset()
    frames = []
    total_reward = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        action_str = {0: "(Left)", 1: "(Down)", 2: "(Right)", 3: "(Up)"}[action]

        # Take step
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Render and print
        rendered = env.render()
        frames.append(rendered)
        print(rendered, action_str)

        state = next_state
        if terminated or truncated:
            break

    # Print final state
    final_render = env.render()
    frames.append(final_render)
    print(final_render)

    return total_reward, frames
