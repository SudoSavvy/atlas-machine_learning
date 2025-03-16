#!/usr/bin/env python3
"""DenseNet-121 Module"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks:

    growth_rate = the growth rate
    compression = the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the keras model

    """
    # init the shape and HeNormal
    init = K.initializers.HeNormal(seed=0)
    input = K.Input(shape=(224, 224, 3))

    # init conv to pooling
    x = K.layers.BatchNormalization(axis=3)(input)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(64, 7,
                        strides=2,
                        padding='same',
                        kernel_initializer=init
                        )(x)
    x = K.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # set filters and layer then dense block and transition
    filters, num_layers = 64, [6, 12, 24, 16]
    for layers in num_layers[:-1]:
        x, filters = dense_block(x, filters, growth_rate, layers)
        x, filters = transition_layer(x, filters, compression)
    x, filters = dense_block(x, filters, growth_rate, num_layers[-1])

    # perform the final process for layers
    x = K.layers.GlobalAveragePooling2D()(x)
    outputs = K.layers.Dense(1000,
                             activation='softmax',
                             kernel_initializer=init
                             )(x)
    keras_model = K.Model(input, outputs)

    # return the keras model
    return (keras_model)