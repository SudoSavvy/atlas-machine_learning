#!/usr/bin/env python3
"""Projection Block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Function that builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015):

    A_prev = the output out of the previous layer
    filters = a tuple or list containing F11, F3, F12, respectively:
    F11 = the number of filters in the first 1x1 convolution
    F3 = the number of filters in the 3x3 convolution
    F12 = the number of filters in the second 1x1 convolution as well
    as the 1x1 convolution in the shortcut connection
    s = the stride of the first convolution in both the main path and
    the shortcut connection
    main path (m_p) = three convolution layers (1x1, 3x3, 1x1)
    shortcut (s_c) = contains a 1x1 convolution to match the dimensions with
    the main output
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the projection block

    """
    # unpack filter tuples
    F11, F3, F12 = filters

    # init using HeNormal
    init = K.initializers.HeNormal(seed=0)

    # main path(m_p) towards the sequence of layers to create new features
    m_p = K.layers.Conv2D(F11, 1,
                          strides=s,
                          padding='same',
                          kernel_initializer=init
                          )(A_prev)
    m_p = K.layers.BatchNormalization()(m_p)
    m_p = K.layers.Activation('relu')(m_p)
    m_p = K.layers.Conv2D(F3, 3,
                          padding='same',
                          kernel_initializer=init
                          )(m_p)
    m_p = K.layers.BatchNormalization()(m_p)
    m_p = K.layers.Activation('relu')(m_p)
    m_p = K.layers.Conv2D(F12, 1,
                          padding='same',
                          kernel_initializer=init
                          )(m_p)
    m_p = K.layers.BatchNormalization()(m_p)

    # shortcut path(s_p) 1x1 to match dimensions
    s_p = K.layers.Conv2D(F12, 1,
                          strides=s,
                          padding='same',
                          kernel_initializer=init
                          )(A_prev)
    shortcut = K.layers.BatchNormalization()(s_p)

    # applying the shortcut to main and apply relu activation
    combine_m_s = K.layers.Add()([m_p, shortcut])
    proj_output = K.layers.Activation('relu')(combine_m_s)

    # returns the activated output of the projection block
    return (proj_output)