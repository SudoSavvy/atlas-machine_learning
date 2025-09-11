# play.py
# Patch keras-rl2 for Keras 3 — MUST be at the top
import tensorflow.keras.models
tensorflow.keras.models.model_from_config = tensorflow.keras.models.model_from_json

# Imports
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
import time

# Environment setup
env = gym.make("ALE/Breakout-v5", render_mode="human")  # Opens a window locally
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
env = FrameStack(env, num_stack=4)

nb_actions = env.action_space.n
input_shape = env.observation_space.shape

# Model setup — must match training
model = Sequential([
    Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
    Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(nb_actions, activation='linear')
])

# DQN Agent setup
memory = SequentialMemory(limit=1000000, window_length=1)  # window_length=1 for playing
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
               nb_steps_warmup=0, target_model_update=1e-2,
               policy=policy, gamma=0.99)

dqn.compile(optimizer=None)
dqn.load_weights("policy.h5")

# Play loop
obs, _ = env.reset()
done = False
while not done:
    action = dqn.forward(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.01)  # small delay to slow down gameplay

env.close()
print("Playback finished!")
