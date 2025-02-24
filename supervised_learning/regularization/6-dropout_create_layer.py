#!/usr/bin/env python3
"""Create a Layer with a Dropout Module"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Function that creates a layer of a neural network using dropout:

    prev = a tensor containing the output of the previous layer
    n = the number of nodes the new layer should contain
    activation = the activation function for the new layer
    keep_prob = the probability that a node will be kept
    training = a boolean indicating whether the model is in training mode

    """
    # init the layer weights using variance scaling
    init_weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode="fan_avg")

    # creates a layer
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=init_weights)

    # apply layer to the prev layer
    layer_out = layer(prev)

    # apply dropout to the layer output
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)
    layer_out = dropout_layer(layer_out, training=training)

    # returns the output
    return (layer_out)
