#!/usr/bin/env python3
"""
Module for loading FrozenLake environments.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from gymnasium.

    Args:
        desc (list[list[str]] | None): Custom map description. If None,
          uses map_name or random.
        map_name (str | None): Pre-made map name (e.g., "4x4", "8x8").
          Ignored if desc is provided.
        is_slippery (bool): Whether the ice is slippery.

    Returns:
        gym.Env: The FrozenLake environment instance.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
