#!/usr/bin/env python3
"""Identity Block for Residual Networks (ResNets)

This module defines the identity block, a key component in building residual 
networks (ResNets), which are widely used for deep learning tasks such as 
image recognition.

The identity block follows the structure outlined in the paper 'Deep Residual 
Learning for Image Recognition' (2015), where the input undergoes multiple 
convolutional layers, batch normalization, and ReLU activation, before being 
added back to the input for the residual connection.

The block consists of the following:
- A 1x1 convolutional layer with 'F11' filters.
- A 3x3 convolutional layer with 'F3' filters.
- A second 1x1 convolutional layer with 'F12' filters.
After each convolution, batch normalization is applied, followed by ReLU 
activation. The final output is the result of adding the residual connection 
and applying a ReLU activation.

Args:
    A_prev (tensorflow.Tensor): The output tensor from the previous layer.
    filters (tuple): A tuple containing the number of filters for each 
    convolutional layer in the block, in the order (F11, F3, F12).

Returns:
    tensorflow.Tensor: The output tensor of the identity block after applying 
    the residual connection and activation.
"""

from tensorflow import keras as K

def identity_block(A_prev, filters):
    """Builds an identity block for a residual network.

    This block consists of two 1x1 convolutions and one 3x3 convolution, 
    with batch normalization and ReLU activation applied after each convolution. 
    The output is added to the original input (identity connection) and passed 
    through a final ReLU activation.

    Args:
        A_prev (tensorflow.Tensor): The output from the previous layer.
        filters (tuple): A tuple containing the number of filters for the 
        convolutions, in the form (F11, F3, F12).

    Returns:
        tensorflow.Tensor: The output tensor of the identity block.
    """
    # Unpack filter values
    F11, F3, F12 = filters

    # Initialize HeNormal initializer
    init = K.initializers.HeNormal(seed=0)

    # First 1x1 convolution
    x1 = K.layers.Conv2D(F11, (1, 1), padding='same', kernel_initializer=init)(A_prev)
    x1 = K.layers.BatchNormalization(axis=-1)(x1)
    x1 = K.layers.Activation('relu')(x1)

    # 3x3 convolution
    x2 = K.layers.Conv2D(F3, (3, 3), padding='same', kernel_init
