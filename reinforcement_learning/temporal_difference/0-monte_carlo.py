#!/usr/bin/env python3
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # handle (obs, info)
            state = state[0]

        episode = []
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = new_state
            if terminated or truncated:
                break

        G = 0
        visited = set()
        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                V[state] += alpha * (G - V[state])

    return V
