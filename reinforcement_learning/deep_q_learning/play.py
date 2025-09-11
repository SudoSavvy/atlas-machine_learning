# reinforcement_learning/deep_q_learning/play.py
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from gymnasium.wrappers import AtariPreprocessing, FrameStack

# Use same image data format as training
keras.backend.set_image_data_format('channels_first')

class GymnasiumToGymWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return obs, reward, done, info

    def render(self, mode='human'):
        try:
            return self.env.render()
        except TypeError:
            return self.env.render(mode=mode)

def make_env(env_id=None):
    candidates = [env_id] if env_id else [
        "ALE/Breakout-v5",
        "BreakoutNoFrameskip-v4",
        "Breakout-v0",
    ]
    last_exc = None
    for cid in candidates:
        if cid is None:
            continue
        try:
            env = gym.make(cid, render_mode="human")  # request human render mode if supported
            print(f"Created environment: {cid}")
            return env
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"Could not create Breakout environment. Last exception: {last_exc}")

def build_model(input_shape, nb_actions, window_length=4):
    # Build same architecture used during training
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(window_length,)+input_shape))
    model.add(layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), activation='relu', data_format='channels_first'))
    model.add(layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), activation='relu', data_format='channels_first'))
    model.add(layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu', data_format='channels_first'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(nb_actions, activation='linear'))
    return model

def main():
    ENV_ID = None
    WINDOW_LENGTH = 4
    WEIGHTS_FILENAME = "policy.h5"

    base_env = make_env(ENV_ID)
    env = AtariPreprocessing(base_env, grayscale_obs=True, scale_obs=False, frame_skip=1, noop_max=30)
    env = FrameStack(env, num_stack=WINDOW_LENGTH)
    env = GymnasiumToGymWrapper(env)

    nb_actions = env.action_space.n
    sample_obs = env.reset()
    obs_shape = sample_obs.shape  # e.g. (4,84,84)
    if len(obs_shape) == 3:
        window_len = obs_shape[0]
        height = obs_shape[1]
        width = obs_shape[2]
    else:
        raise RuntimeError(f"Unexpected observation shape: {obs_shape}")

    # Build the model (architecture must match training)
    model = build_model(input_shape=(height, width), nb_actions=nb_actions, window_length=WINDOW_LENGTH)

    # Memory & Greedy policy for play
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = GreedyQPolicy()  # greedy during play

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=0,
                   target_model_update=1e-3, policy=policy)
    dqn.compile(keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae'])

    # Load weights saved by train.py
    print(f"Loading weights from {WEIGHTS_FILENAME} ...")
    dqn.load_weights(WEIGHTS_FILENAME)

    # Run episodes and visualize
    print("Running test episodes with GreedyQPolicy...")
    # nb_episodes controls how many games to play; set to desired number
    dqn.test(env, nb_episodes=5, visualize=True)

if __name__ == "__main__":
    main()
