#!/usr/bin/env python3
"""ResNet-50 Module"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed by
    batch normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the keras model

    """
    # init the HeNormal and input for shape
    input = K.Input(shape=(224, 224, 3))
    init = K.initializers.HeNormal(seed=0)

    # init conv and pooling
    layer_output = K.layers.Conv2D(64,
                                   (7, 7),
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=init
                                   )(input)
    layer_output = K.layers.BatchNormalization(axis=3)(layer_output)
    layer_output = K.layers.Activation('relu')(layer_output)
    layer_output = K.layers.MaxPooling2D((3, 3),
                                         strides=2,
                                         padding='same'
                                         )(layer_output)

    # the ResNet stages == conv2_x
    layer_output = projection_block(layer_output, [64, 64, 256], s=1)
    layer_output = identity_block(layer_output, [64, 64, 256])
    layer_output = identity_block(layer_output, [64, 64, 256])

    # conv3_x
    layer_output = projection_block(layer_output, [128, 128, 512], s=2)
    layer_output = identity_block(layer_output, [128, 128, 512])
    layer_output = identity_block(layer_output, [128, 128, 512])
    layer_output = identity_block(layer_output, [128, 128, 512])

    # conv4_x
    layer_output = projection_block(layer_output, [256, 256, 1024], s=2)
    layer_output = identity_block(layer_output, [256, 256, 1024])
    layer_output = identity_block(layer_output, [256, 256, 1024])
    layer_output = identity_block(layer_output, [256, 256, 1024])
    layer_output = identity_block(layer_output, [256, 256, 1024])
    layer_output = identity_block(layer_output, [256, 256, 1024])

    # conv5_x
    layer_output = projection_block(layer_output, [512, 512, 2048], s=2)
    layer_output = identity_block(layer_output, [512, 512, 2048])
    layer_output = identity_block(layer_output, [512, 512, 2048])

    # average the pooling and and connect the layer
    layer_output = K.layers.GlobalAveragePooling2D()(layer_output)
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init
                            )(layer_output)

    # create the model
    keras_model = K.Model(inputs=input, outputs=output)

    # returns the keras model
    return (keras_model)