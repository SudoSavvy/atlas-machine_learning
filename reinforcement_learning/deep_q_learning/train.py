# train.py
# Patch keras-rl2 for Keras 3 — MUST be at the top
import tensorflow.keras.models
tensorflow.keras.models.model_from_config = tensorflow.keras.models.model_from_json

# Imports
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Environment setup
env = gym.make("ALE/Breakout-v5", render_mode=None)  # No rendering during training
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
env = FrameStack(env, num_stack=4)

nb_actions = env.action_space.n
input_shape = env.observation_space.shape

# Model setup
model = Sequential([
    Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
    Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(nb_actions, activation='linear')
])

# DQN Agent setup
memory = SequentialMemory(limit=1000000, window_length=4)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=1000, target_model_update=1e-2,
               policy=policy, gamma=0.99, train_interval=4, delta_clip=1.0)

dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

# Train — for testing, you can use a small number of steps
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

# Save weights
dqn.save_weights("policy.h5", overwrite=True)

env.close()
print("Training complete! Weights saved to policy.h5")
