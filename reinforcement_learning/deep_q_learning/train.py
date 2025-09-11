# reinforcement_learning/deep_q_learning/train.py
import os
import gymnasium as gym
import numpy as np

# Keras and keras-rl2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

# Gymnasium wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStack

# Make Keras use 'channels_first' format for compatibility with keras-rl conv input examples
keras.backend.set_image_data_format('channels_first')

class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Wrap a gymnasium.Env to behave like old gym.Env for keras-rl compatibility:
      - reset(...) returns observation (not (obs, info))
      - step(action) returns (obs, reward, done, info) instead of (obs, reward, terminated, truncated, info)
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return obs, reward, done, info

    def render(self, mode='human'):
        # delegate to underlying env
        try:
            return self.env.render()
        except TypeError:
            # some gymnasium envs accept a mode argument
            return self.env.render(mode=mode)

def make_env(env_id=None):
    # Try a set of common Breakout environment ids; pick first that works
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
            env = gym.make(cid, render_mode=None)  # render handled later; render_mode=None keeps compatibility
            print(f"Created environment: {cid}")
            return env
        except Exception as e:
            last_exc = e
            # continue to next candidate
    raise RuntimeError(f"Could not create Breakout environment. Last exception: {last_exc}")

def build_model(input_shape, nb_actions, window_length=4):
    # input_shape here is the (height, width) after AtariPreprocessing (84x84)
    # keras-rl expects input_shape = (window_length, height, width) if channels_first
    model = keras.models.Sequential()
    model.add(layers.InputLayer(input_shape=(window_length,)+input_shape))  # channels_first: (4,84,84)
    # Convolutional layers similar to DeepMind DQN
    model.add(layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', data_format='channels_first'))
    model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', data_format='channels_first'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_first'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(nb_actions, activation='linear'))  # Q-values for each action
    print(model.summary())
    return model

def main():
    # hyperparameters (tweak as needed)
    ENV_ID = None  # or set to a specific env id string
    WINDOW_LENGTH = 4
    TRAIN_STEPS = 200000  # reduce/increase depending on runtime
    MEMORY_LIMIT = 1000000
    WARMUP_STEPS = 50000
    TARGET_MODEL_UPDATE = 10000
    LR = 0.00025
    WEIGHTS_FILENAME = "policy.h5"

    # Create environment
    base_env = make_env(ENV_ID)
    # AtariPreprocessing -> grayscale, resize to 84x84, optional frame_skip.
    env = AtariPreprocessing(base_env, grayscale_obs=True, scale_obs=False, frame_skip=1, noop_max=30)
    # FrameStack to produce stacked frames shape (4,84,84) channels_first
    env = FrameStack(env, num_stack=WINDOW_LENGTH)
    # Compatibility wrapper for keras-rl (old Gym API)
    env = GymnasiumToGymWrapper(env)

    # Determine the action space and observation shape
    nb_actions = env.action_space.n
    # Observation space shape after AtariPreprocessing + FrameStack (channels_first)
    # gymnasium FrameStack returns array shape (num_stack, H, W)
    sample_obs = env.reset()
    # sample_obs could be a numpy array already
    obs_shape = sample_obs.shape  # e.g. (4,84,84)
    # For model builder we want input (height, width) (excluding window length)
    # since build_model expects input_shape=(height,width) and will add window_length
    if len(obs_shape) == 3:
        window_len = obs_shape[0]
        height = obs_shape[1]
        width = obs_shape[2]
    else:
        raise RuntimeError(f"Unexpected observation shape: {obs_shape}")

    model = build_model(input_shape=(height, width), nb_actions=nb_actions, window_length=WINDOW_LENGTH)

    # Memory, policy, agent
    memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGTH)
    policy = EpsGreedyQPolicy()  # epsilon-greedy during training
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=WARMUP_STEPS,
                   target_model_update=TARGET_MODEL_UPDATE, policy=policy, enable_double_dqn=True,
                   enable_dueling_network=False)
    dqn.compile(keras.optimizers.Adam(learning_rate=LR), metrics=['mae'])

    # Fit the agent
    print("Starting training...")
    dqn.fit(env, nb_steps=TRAIN_STEPS, visualize=False, verbose=2)

    # Save weights (keras-rl2 style)
    print(f"Saving agent weights to {WEIGHTS_FILENAME} ...")
    dqn.save_weights(WEIGHTS_FILENAME, overwrite=True)

    # Optionally: save the full model too (not necessary for keras-rl2 load_weights)
    model_save_path = "dqn_model_full.h5"
    print(f"Also saving full Keras model to {model_save_path} ...")
    model.save(model_save_path, include_optimizer=False)

    print("Training complete.")

if __name__ == "__main__":
    main()
