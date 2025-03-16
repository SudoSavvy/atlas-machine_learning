#!/usr/bin/env python3
"""Transition Layer Module"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a transition layer as described in
    Densely Connected Convolutional Networks:

    X = the output from the previous layer
    nb_filters = an integer representing the number of filters in X
    compression = the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of filters
    within the output, respectively

    """
    # init HeNormal, calculate compressed filters, set input
    init = K.initializers.HeNormal(seed=0)
    filters = int(nb_filters * compression)
    input = X

    # set output to batch norm, relu, 1x1 conv, and 2x2 avg pooling
    trans_out = K.layers.BatchNormalization()(input)
    trans_out = K.layers.Activation('relu')(trans_out)
    trans_out = K.layers.Conv2D(filters,
                                (1, 1),
                                padding='same',
                                kernel_initializer=init
                                )(trans_out)
    trans_out = K.layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='same'
                                          )(trans_out)

    # returns the output of the transition layer, and filters
    return (trans_out, filters)