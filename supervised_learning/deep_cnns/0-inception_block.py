#!/usr/bin/env python3
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described
    in Going Deeper with Convolutions (2014).

    Parameters:
    - A_prev: the output from the previous layer
    - filters: tuple containing (F1, F3R, F3, F5R, F5, FPP):
        * F1: filters in the 1x1 convolution
        * F3R: filters in the 1x1 convolution before the 3x3 convolution
        * F3: filters in the 3x3 convolution
        * F5R: filters in the 1x1 convolution before the 5x5 convolution
        * F5: filters in the 5x5 convolution
        * FPP: filters in the 1x1 convolution after max pooling

    Returns:
    - The concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 Convolution branch
    conv1x1 = K.layers.Conv2D(
        F1, (1, 1), activation="relu", padding="same"
    )(A_prev)

    # 1x1 Convolution before 3x3 Convolution branch
    conv3x3_reduce = K.layers.Conv2D(
        F3R, (1, 1), activation="relu", padding="same"
    )(A_prev)
    conv3x3 = K.layers.Conv2D(
        F3, (3, 3), activation="relu", padding="same"
    )(conv3x3_reduce)

    # 1x1 Convolution before 5x5 Convolution branch
    conv5x5_reduce = K.layers.Conv2D(
        F5R, (1, 1), activation="relu", padding="same"
    )(A_prev)
    conv5x5 = K.layers.Conv2D(
        F5, (5, 5), activation="relu", padding="same"
    )(conv5x5_reduce)

    # Max pooling branch followed by 1x1 Convolution
    maxpool = K.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding="same"
    )(A_prev)
    conv_maxpool = K.layers.Conv2D(
        FPP, (1, 1), activation="relu", padding="same"
    )(maxpool)

    # Concatenate all branches
    output = K.layers.Concatenate()(
        [conv1x1, conv3x3, conv5x5, conv_maxpool]
    )

    return output
