#!/usr/bin/env python3
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains a policy using REINFORCE with the given environment.

    env: the initial environment
    nb_episodes: number of episodes used for training
    alpha: learning rate
    gamma: discount factor
    Returns: a list containing the score for each episode
    """
    # Extract input and output dimensions from environment
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize theta (policy parameters)
    theta = np.random.rand(n_obs, n_actions)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0

        # Generate an episode
        done = False
        while not done:
            probs = softmax(np.dot(state, theta))
            action = np.random.choice(n_actions, p=probs)
            next_state, reward, done, truncated, _ = env.step(action)

            # Get gradient from policy_gradient
            grad = policy_gradient(state, theta, action)
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state

        # Compute discounted rewards
        G = np.zeros_like(rewards)
        running_sum = 0
        for t in reversed(range(len(rewards))):
            running_sum = running_sum * gamma + rewards[t]
            G[t] = running_sum
        G = (G - np.mean(G)) / (np.std(G) + 1e-8)

        # Update theta
        for t in range(len(grads)):
            theta += alpha * grads[t] * G[t]

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

    return scores


def softmax(x):
    """Numerically stable softmax"""
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
