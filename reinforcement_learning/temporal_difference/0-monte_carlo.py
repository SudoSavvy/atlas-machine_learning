#!/usr/bin/env python3
import numpy as np

# Initialize value function
V = np.zeros(64)  # 8x8 FrozenLake has 64 states

# Define a simple deterministic policy (e.g., always go right)
def policy(state):
    return 2  # action 2 = RIGHT

# Import your Monte Carlo function
from monte_carlo import monte_carlo

# Run Monte Carlo value estimation
V = monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)

# Reshape and print the value function as a grid
print(np.round(V.reshape((8, 8)), 4))
