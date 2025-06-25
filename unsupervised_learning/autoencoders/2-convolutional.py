#!/usr/bin/env python3
"""Convolutional autoencoder network definition"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input (H, W, C).
        filters (list): List of integers containing the number of filters
                        for each convolutional layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model, compiled.
    """
    # Encoder
    input_layer = keras.Input(shape=input_dims)
    x = input_layer
    for f in filters:
        x = keras.layers.Conv2D(
            filters=f, kernel_size=(3, 3), padding='same',
            activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    encoder = keras.Model(inputs=input_layer, outputs=x)

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input
    for i, f in enumerate(reversed(filters)):
        # Last two layers handled separately below
        if i < len(filters) - 2:
            x = keras.layers.Conv2D(
                filters=f, kernel_size=(3, 3), padding='same',
                activation='relu')(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        elif i == len(filters) - 2:
            x = keras.layers.Conv2D(
                filters=f, kernel_size=(3, 3), padding='valid',
                activation='relu')(x)
        else:
            x = keras.layers.Conv2D(
                filters=input_dims[2], kernel_size=(3, 3), padding='same',
                activation='sigmoid')(x)

    decoder = keras.Model(inputs=decoder_input, outputs=x)

    # Autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
