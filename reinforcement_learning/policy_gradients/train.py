#!/usr/bin/env python3
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements a Monte-Carlo policy gradient (REINFORCE) training loop.

    Args:
        env: OpenAI Gym-like environment
        nb_episodes (int): number of episodes for training
        alpha (float): learning rate
        gamma (float): discount factor

    Returns:
        list: total rewards (scores) for each episode
    """
    # Initialize random weights: (obs_space, action_space)
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)

    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        score = 0

        # Run an episode
        while not done:
            action, grad = policy_gradient(state, weight)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(grad)
            episode_rewards.append(reward)
            score += reward
            state = next_state

        # Compute returns (G) for each step
        G = 0
        returns = []
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)

        # Normalize returns for stability
        if np.std(returns) > 0:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        # Update weights using gradients
        for grad, Gt in zip(episode_actions, returns):
            weight += alpha * grad * Gt

        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores
