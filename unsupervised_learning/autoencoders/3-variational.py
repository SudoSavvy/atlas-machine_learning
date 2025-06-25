#!/usr/bin/env python3
"""Variational autoencoder network definition"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): List of nodes for each hidden encoder layer.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): Outputs z, z_mean, z_log_var
        decoder (keras.Model): Decoder model
        auto (keras.Model): Full autoencoder model, compiled
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0], latent_dims))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(inputs=inputs, outputs=[z, z_mean, z_log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Autoencoder
    z, z_mean, z_log_var = encoder(inputs)
    reconstructed = decoder(z)
    auto = keras.Model(inputs=inputs, outputs=reconstructed)

    # Loss
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, reconstructed)
    reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=1)

    kl_loss = -0.5 * keras.backend.sum(1 + z_log_var -
                                       keras.backend.square(z_mean) -
                                       keras.backend.exp(z_log_var), axis=1)

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
