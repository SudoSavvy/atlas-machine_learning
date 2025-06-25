#!/usr/bin/env python3
"""Convolutional autoencoder network definition"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (H, W, C).
        filters (list): List of integers for each Conv2D layer in encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): Encoder model.
        decoder (keras.Model): Decoder model.
        auto (keras.Model): Autoencoder model, compiled.
    """
    # Encoder
    input_layer = keras.Input(shape=input_dims)
    x = input_layer
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs=input_layer, outputs=x)

    # Decoder
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    # Second-to-last layer: match original first encoder filter
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu',
                            padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    # Output layer
    x = keras.layers.Conv2D(input_dims[2], (3, 3), activation='sigmoid',
                            padding='same')(x)
    decoder = keras.Model(inputs=latent_input, outputs=x)

    # Autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
