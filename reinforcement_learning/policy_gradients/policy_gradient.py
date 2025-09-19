#!/usr/bin/env python3
import numpy as np


def policy(state, weight):
    """
    Computes a softmax policy given a state and a weight matrix.

    Args:
        state (np.ndarray): shape (1, n), current state (row vector)
        weight (np.ndarray): shape (n, m), weight matrix

    Returns:
        np.ndarray: shape (1, m), probabilities for each action
    """
    z = np.matmul(state, weight)
    exp = np.exp(z - np.max(z))  # numerical stability
    return exp / np.sum(exp, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient.

    Args:
        state (np.ndarray): shape (n,), current observation
        weight (np.ndarray): shape (n, m), weight matrix

    Returns:
        action (int): chosen action index
        gradient (np.ndarray): gradient of log-probability, shape (n, m)
    """
    # Ensure state is 2D row vector
    state = state.reshape(1, -1)

    # Get action probabilities
    probs = policy(state, weight)

    # Sample action based on probabilities
    action = np.random.choice(probs.shape[1], p=probs[0])

    # One-hot encode chosen action
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1

    # Gradient: state.T * (one_hot - probs)
    grad = np.matmul(state.T, one_hot - probs)

    return action, grad
