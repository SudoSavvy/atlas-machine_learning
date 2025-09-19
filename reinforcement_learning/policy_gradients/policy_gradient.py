#!/usr/bin/env python3
import numpy as np

policy = __import__('policy_gradient').policy  # reuse the previous policy function


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient.

    Args:
        state (np.ndarray): shape (4,), current observation of the environment
        weight (np.ndarray): shape (4, 2), weight matrix

    Returns:
        action (int): chosen action (0 or 1)
        gradient (np.ndarray): same shape as weight, gradient of log-probability
    """
    # Ensure state is a row vector
    state = state.reshape(1, -1)

    # Compute probabilities using the policy
    probs = policy(state, weight)

    # Choose action based on probabilities
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Compute gradient of log π(a|s)
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1

    grad = np.matmul(state.T, (one_hot - probs))

    return action, grad
