#!/usr/bin/env python3
"""Builds the ResNet-50 architecture"""

from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture"""

    model_start = K.layers.Input(shape=(224, 224, 3))

    # Start of stage 1
    layers = K.layers.Conv2D(
        kernel_size=7,
        padding="same",
        filters=64,
        strides=2,
        kernel_initializer=K.initializers.he_normal(),
    )(model_start)

    layers = K.layers.BatchNormalization(axis=3)(layers)
    layers = K.layers.Activation("relu")(layers)

    layers = K.layers.MaxPooling2D(pool_size=3,
                                   strides=2,
                                   padding="same")(layers)

    # Start of stage 2
    layers = projection_block(layers, [64, 64, 256], 1)
    layers = identity_block(layers, [64, 64, 256])
    layers = identity_block(layers, [64, 64, 256])

    # start of stage 3
    layers = projection_block(layers, [128, 128, 512], 2)
    layers = identity_block(layers, [128, 128, 512])
    layers = identity_block(layers, [128, 128, 512])
    layers = identity_block(layers, [128, 128, 512])

    # start of stage 4
    layers = projection_block(layers, [256, 256, 1024], 2)
    layers = identity_block(layers, [256, 256, 1024])
    layers = identity_block(layers, [256, 256, 1024])
    layers = identity_block(layers, [256, 256, 1024])
    layers = identity_block(layers, [256, 256, 1024])
    layers = identity_block(layers, [256, 256, 1024])

    # start of staage 5
    layers = projection_block(layers, [512, 512, 2048], 2)
    layers = identity_block(layers, [512, 512, 2048])
    layers = identity_block(layers, [512, 512, 2048])

    # last stage
    # Need to find out why the parameters are this way
    layers = K.layers.AveragePooling2D(pool_size=7,
                                       strides=1,
                                       padding="valid")(layers)

    # layers = K.layers.Flatten()(layers)

    layers = K.layers.Dense(
        1000, activation="softmax",
        kernel_initializer=K.initializers.he_normal()
    )(layers)

    model = K.models.Model(model_start, layers)

    return model