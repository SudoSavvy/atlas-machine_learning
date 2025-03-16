#!/usr/bin/env python3
"""Inception Network Builder

This script defines a function `inception_network` that constructs the Inception Network 
as described in the 2014 paper 'Going Deeper with Convolutions.' The architecture 
includes a series of convolutional layers and Inception blocks to perform image classification tasks.

The input to the model is expected to have the shape (224, 224, 3), which corresponds 
to an image of size 224x224 with 3 color channels (RGB). All convolution layers 
are followed by a Rectified Linear Unit (ReLU) activation function to introduce non-linearity.

The network utilizes multiple Inception blocks with different filter sizes, which allows 
the model to capture information at various levels of abstraction.

Returns:
    keras.Model: A Keras model representing the complete Inception Network for image classification.
"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block

def inception_network():
    """Constructs the Inception Network model.

    This function builds the Inception Network as defined in the 'Going Deeper with Convolutions' 
    paper (2014). The model consists of convolutional layers followed by multiple Inception blocks, 
    which are designed to process images at various scales. The network ends with average pooling 
    and a fully connected layer for classification.

    The input shape for the model is (224, 224, 3), which corresponds to a 224x224 RGB image.

    The model architecture follows the structure described in the paper, with the following stages:
    1. Initial convolutional layers for feature extraction.
    2. Multiple Inception blocks to capture multi-scale features.
    3. Final average pooling and fully connected layers for classification.

    Returns:
        keras.Model: The constructed Keras model representing the Inception Network.
    """
    # Define the input layer with the specified input shape (224, 224, 3)
    input = K.Input(shape=(224, 224, 3))

    # Initial Convolutional Layer: Applies a 7x7 convolution with 64 filters and ReLU activation
    x = K.layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(input)

    # MaxPooling Layer: Downsamples the feature map using 3x3 pooling with strides of 2
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Second Convolutional Layer: 1x1 convolution with 64 filters and ReLU activation
    x = K.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)

    # Third Convolutional Layer: 3x3 convolution with 192 filters and ReLU activation
    x = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)

    # MaxPooling Layer: Downsamples the feature map using 3x3 pooling with strides of 2
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Apply multiple Inception blocks to capture features at different scales
    incept_3a = inception_block(x, [64, 96, 128, 16, 32, 32])
    incept_3b = inception_block(incept_3a, [128, 128, 192, 32, 96, 64])

    # MaxPooling Layer: Downsamples the feature map using 3x3 pooling with strides of 2
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(incept_3b)

    # Apply more Inception blocks for deeper feature extraction
    incept_4a = inception_block(x, [192, 96, 208, 16, 48, 64])
    incept_4b = inception_block(incept_4a, [160, 112, 224, 24, 64, 64])
    incept_4c = inception_block(incept_4b, [128, 128, 256, 24, 64, 64])
    incept_4d = inception_block(incept_4c, [112, 144, 288, 32, 64, 64])
    incept_4e = inception_block(incept_4d, [256, 160, 320, 32, 128, 128])

    # MaxPooling Layer: Downsamples the feature map using 3x3 pooling with strides of 2
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(incept_4e)

    # Apply the final Inception blocks to further refine the features
    incept_5a = inception_block(x, [256, 160, 320, 32, 128, 128])
    incept_5b = inception_block(incept_5a, [384, 192, 384, 48, 128, 128])

    # Apply average pooling to reduce the spatial dimensions of the feature map
    x = K.layers.AveragePooling2D(pool_size=(7, 7))(incept_5b)

    # Dropout Layer: Applies dropout with a rate of 0.4 to reduce overfitting
    x = K.layers.Dropout(0.4)(x)

    # Fully Connected Layer: Dense layer with 1000 units and softmax activation for classification
    output = K.layers.Dense(1000, activation='softmax')(x)

    # Define the model using the input and output layers
    keras_model = K.Model(input, output)

    # Return the constructed Keras model
    return keras_model
