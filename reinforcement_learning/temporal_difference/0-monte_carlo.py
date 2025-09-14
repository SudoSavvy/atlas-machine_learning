#!/usr/bin/env python3
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.9):
    """
    Performs first-visit Monte Carlo value estimation.

    env: the environment instance
    V: numpy.ndarray of shape (s,) value estimates
    policy: function that maps state -> action
    episodes: number of episodes to sample
    max_steps: maximum steps per episode
    alpha: ignored here (we use full return)
    gamma: discount factor
    """
    for _ in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):  # gymnasium reset returns (obs, info)
            state = state[0]

        episode = []
        for _ in range(max_steps):
            action = policy(state)
            step_out = env.step(action)
            if len(step_out) == 4:  # gym
                new_state, reward, done, _ = step_out
            else:  # gymnasium
                new_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            if isinstance(new_state, tuple):
                new_state = new_state[0]
            episode.append((state, reward))
            state = new_state
            if done:
                break

        G = 0
        visited = set()
        for s, r in reversed(episode):
            G = r + gamma * G
            if s not in visited:
                visited.add(s)
                V[s] = G  # full MC return, no incremental averaging

    return V
