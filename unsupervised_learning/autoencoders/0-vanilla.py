#!/usr/bin/env python3
"""Autoencoder network definition"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder neural network using Keras.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): List of integers containing the number of nodes
                              for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model, compiled.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(units=nodes, activation='relu')(x)
    latent = keras.layers.Dense(units=latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=inputs, outputs=latent)

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(units=nodes, activation='relu')(x)
    outputs = keras.layers.Dense(units=input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded)

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
