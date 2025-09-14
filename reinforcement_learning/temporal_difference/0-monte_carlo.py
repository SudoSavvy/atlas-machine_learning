#!/usr/bin/env python3

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    
    returns = {state: [] for state in range(episodes)}

    for i in range(episodes):
        state = env.reset()[0]
        episode = []
        terminated = False
        truncated = False
        step_counter = 0

        while not terminated and not truncated:

            action = policy(state)

            new_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, reward))
            if terminated:
                break
            state = new_state

        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            returns[state].append(G)
            V[state] = np.mean(returns[state])

    return V