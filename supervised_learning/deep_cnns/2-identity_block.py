#!/usr/bin/env python3
"""Identity Block Module"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Function that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015):

    A_prev = the output out of the previous layer
    filters = a tuple or list containing F11, F3, F12, respectively:
    F11 = the number of filters in the first 1x1 convolution
    F3 = the number of filters in the 3x3 convolution
    F12 = the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the identity block

    """
    # unpack filter tuples
    F11, F3, F12 = filters

    # init using HeNormal
    init = K.initializers.HeNormal(seed=0)

    # 1x1 convolution (the first)
    x1 = K.layers.Conv2D(F11, (1, 1),
                         padding='same',
                         kernel_initializer=init
                         )(A_prev)
    x1 = K.layers.BatchNormalization(axis=-1)(x1)
    x1 = K.layers.Activation('relu')(x1)

    # 3x3 convolution
    x2 = K.layers.Conv2D(F3, (3, 3),
                         padding='same',
                         kernel_initializer=init
                         )(x1)
    x2 = K.layers.BatchNormalization(axis=-1)(x2)
    x2 = K.layers.Activation('relu')(x2)

    # 1x1 convolution (the second)
    x3 = K.layers.Conv2D(F12, (1, 1),
                         padding='same',
                         kernel_initializer=init
                         )(x2)
    x3 = K.layers.BatchNormalization(axis=-1)(x3)

    # add identity connection and apply activation
    add_output = K.layers.Add()([x3, A_prev])
    id_output = K.layers.Activation('relu')(add_output)

    # returns the activated output of the identity block
    return (id_output)