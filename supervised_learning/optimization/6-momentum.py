#!/usr/bin/env python3

import tensorflow as tf

def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm in TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight.

    Returns:
    tf.keras.optimizers.Optimizer: A TensorFlow optimizer using momentum.
    """
    return tf.kera.optimizers.SGD(learning_rate=alpha, momentum=beta1)
