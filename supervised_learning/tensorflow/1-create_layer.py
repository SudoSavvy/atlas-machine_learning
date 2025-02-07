#!/usr/bin/env python3
"""Module that creates a layer for a neural network using TensorFlow v1"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network using TensorFlow v1.

    Args:
        prev (tf.Tensor): The tensor output from the previous layer.
        n (int): The number of nodes in the layer to create.
        activation (function): The activation function to be used in the
        layer.

    Returns:
        tf.Tensor: The tensor output of the layer.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name="layer")
    return layer(prev)
