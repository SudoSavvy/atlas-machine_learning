#!/usr/bin/env python3

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tf.Tensor): Activated output of the previous layer.
        n (int): Number of nodes in the layer to be created.
        activation (callable): Activation function to apply to the output.

    Returns:
        tf.Tensor: Activated output for the batch-normalized layer.
    """
    # Define the Dense layer with VarianceScaling initializer
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )

    # Apply the Dense layer to the previous layer output
    Z = dense_layer(prev)

    # Calculate mean and variance for batch normalization
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Initialize gamma and beta as trainable variables
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Perform batch normalization
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-7)

    # Apply activation function
    if activation is not None:
        return activation(Z_norm)
    return Z_norm
