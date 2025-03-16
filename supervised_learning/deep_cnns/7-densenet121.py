#!/usr/bin/env python3
"""
DenseNet-121
"""

# Import Keras library as K
import tensorflow.keras as K

# Import the necessary functions
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Function to build a DenseNet-121 architecture
    """
    # Initialize weights with He normal distribution
    initializer = K.initializers.he_normal()

    # Define the input layer shape
    X = K.Input(shape=(224, 224, 3))

    # Normalize input
    norm_1 = K.layers.BatchNormalization()
    output_1 = norm_1(X)

    # Apply ReLU activation to the normalized input
    activ_1 = K.layers.Activation('relu')
    output_1 = activ_1(output_1)

    # Apply first convolutional operation with 64 filters and a 7x7 kernel
    layer_1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              padding='same',
                              strides=2,
                              kernel_initializer=initializer,
                              activation=None)
    output_1 = layer_1(output_1)

    # Apply max pooling operation with a 3x3 window
    layer_2 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_2 = layer_2(output_1)

    # Add the first dense block with 6 layers
    db1_output = dense_block(output_2, output_2.shape[-1], growth_rate, 6)

    # Add the first transition layer with compression
    tl1_output = transition_layer(db1_output[0],
                                  int(db1_output[1]), compression)

    # Add the second dense block with 12 layers
    db2_output = dense_block(tl1_output[0], tl1_output[1], growth_rate, 12)

    # Add the second transition layer with compression
    tl2_output = transition_layer(db2_output[0],
                                  int(db2_output[1]), compression)

    # Add the third dense block with 24 layers
    db3_output = dense_block(tl2_output[0], tl2_output[1], growth_rate, 24)

    # Add the third transition layer with compression
    tl3_output = transition_layer(db3_output[0],
                                  int(db3_output[1]), compression)

    # Add the fourth dense block with 16 layers
    db4_output = dense_block(tl3_output[0], tl3_output[1], growth_rate, 16)

    # Apply average pooling with a 7x7 window
    layer_3 = K.layers.AvgPool2D(pool_size=7,
                                 padding='same',
                                 strides=None)
    output_3 = layer_3(db4_output[0])

    # Apply a dense layer with softmax activation for final classification
    softmax = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    output_4 = softmax(output_3)

    # Define the model with input and output layers
    model = K.models.Model(inputs=X, outputs=output_4)

    return model
