#!/usr/bin/env python3
"""L2 Regularization Layer Module"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a neural network layer in tensorFlow that
    includes L2 regularization:

    prev = a tensor containing the output of the previous layer
    n = the number of nodes the new layer should contain
    activation = the activation function that should be used on the layer
    lambtha = the L2 regularization parameter

    """
    # initialize weights with variance scaling
    init_weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode=("fan_avg"))
    # applying the L2 regularization
    l2_reg = tf.keras.regularizers.L2(lambtha)

    # creating a layer with the activation and regularization
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=init_weights,
                                  kernel_regularizer=l2_reg)

    # returns the output of the new layer
    return layer(prev)
