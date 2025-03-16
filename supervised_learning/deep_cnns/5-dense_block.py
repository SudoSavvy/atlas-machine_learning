#!/usr/bin/env python3
"""Dense Block Module"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block as described in Densely
    Connected Convolutional Networks:

    X = the output from the previous layer
    nb_filters = an integer representing the number of filters in X
    growth_rate = the growth rate for the dense block
    layers = the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively

    """
    # init HeNormal and set output to X
    init = K.initializers.HeNormal(seed=0)
    output = X

    # iterate over the range of layers and creates a new layer
    for _ in range(layers):
        # bottleneck layers (bl)
        bl = K.layers.BatchNormalization()(output)
        bl = K.layers.Activation('relu')(bl)
        bl = K.layers.Conv2D(4 * growth_rate, 1, 1,
                             padding='same',
                             kernel_initializer=init
                             )(bl)

        # 3x3 convolution
        conv_out = K.layers.BatchNormalization()(bl)
        conv_out = K.layers.Activation('relu')(conv_out)
        conv_out = K.layers.Conv2D(growth_rate, (3, 3),
                                   padding='same',
                                   kernel_initializer=init
                                   )(conv_out)

        # concatenate with output
        output = K.layers.Concatenate()([output, conv_out])
        nb_filters += growth_rate

    # returns the concatenate output and number of filters
    return (output, nb_filters)