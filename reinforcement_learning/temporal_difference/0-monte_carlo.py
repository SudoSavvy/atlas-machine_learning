#!/usr/bin/env python3
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.9):
    """
    First-visit Monte Carlo prediction with incremental updates.
    """
    for _ in range(episodes):
        state = env.reset()
        # gymnasium returns (obs, info) tuple, gym returns just obs
        if isinstance(state, tuple):
            state = state[0]

        episode = []
        for _ in range(max_steps):
            action = policy(state)
            step = env.step(action)

            # Handle both gymnasium and gym return signatures
            if len(step) == 5:
                new_state, reward, terminated, truncated, _ = step
            else:  # older gym
                new_state, reward, done, _ = step
                terminated, truncated = done, False

            episode.append((state, reward))
            state = new_state
            if terminated or truncated:
                break

        # Monte Carlo return calculation (first-visit)
        G = 0
        visited = set()
        for s, reward in reversed(episode):
            G = reward + gamma * G
            if s not in visited:
                visited.add(s)
                V[s] += alpha * (G - V[s])

    return V
