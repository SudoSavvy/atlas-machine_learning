#!/usr/bin/env python3
import tensorflow as tf

def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    
    Parameters:
    prev (tf.Tensor): Tensor containing the output of the previous layer
    n (int): Number of nodes the new layer should contain
    activation (function): Activation function for the new layer
    keep_prob (float): Probability that a node will be kept
    training (bool): Whether the model is in training mode
    
    Returns:
    tf.Tensor: Output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, activation=None, kernel_initializer=initializer)(prev)
    if training:
        layer = tf.keras.layers.Dropout(rate=1 - keep_prob)(layer, training=True)
    return activation(layer)
