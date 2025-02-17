#!/usr/bin/env python3

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The weight used to determine the rate of decay.
        decay_step (int): The number of steps before applying decay.

    Returns:
        tf.Tensor: A TensorFlow operation representing the decayed learning
        rate.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
