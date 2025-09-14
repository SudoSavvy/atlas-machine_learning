#!/usr/bin/env python3
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to update the value function V.

    Args:
        env: the environment instance
        V (np.ndarray): shape (s,) containing the value estimates
        policy (function): function that takes in a state and returns an action
        episodes (int): total number of episodes to run
        max_steps (int): maximum number of steps per episode
        alpha (float): learning rate
        gamma (float): discount factor

    Returns:
        np.ndarray: the updated value estimates
    """
    for _ in range(episodes):
        state = env.reset()
        episode = []  # stores (state, reward) pairs

        # Generate one episode following the policy
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            state = new_state
            if done:
                break

        # Compute returns G and update V for each state visited
        G = 0
        visited = set()
        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state not in visited:
                visited.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
