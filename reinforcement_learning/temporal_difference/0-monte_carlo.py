#!/usr/bin/env python3
import numpy as np

# Initialize value function for FrozenLake8x8 (64 states)
V = np.zeros(64)

# Define a simple deterministic policy: always move RIGHT
def policy(state):
    return 2  # action 2 corresponds to RIGHT

# Import your Monte Carlo function from monte_carlo.py
from monte_carlo import monte_carlo

# Run Monte Carlo value estimation
V = monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)

# Print the value function reshaped as an 8x8 grid
print(np.round(V.reshape((8, 8)), 4))
