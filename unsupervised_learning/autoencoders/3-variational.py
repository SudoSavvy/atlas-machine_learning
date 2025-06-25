#!/usr/bin/env python3
"""Variational autoencoder network definition"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): List of integers containing the number of
                              nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        encoder (keras.Model): The encoder model, outputs z, mean, log_var.
        decoder (keras.Model): The decoder model.
        auto (keras.Model): The full autoencoder model, compiled.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Mean and log variance layers
    z_mean = keras.layers.Dense(latent_dims)(x)
    z_log_var = keras.layers.Dense(latent_dims)(x)

    # Sampling layer using reparameterization trick
    def sampling(args):
        mean, log_var = args
        epsilon = keras.backend.random_normal(shape=(keras.backend.shape(mean)[0], latent_dims))
        return mean + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    encoder = keras.Model(inputs=inputs, outputs=[z, z_mean, z_log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=output)

    # Autoencoder
    vae_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs=inputs, outputs=vae_outputs)

    # Loss function: reconstruction loss + KL divergence
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, vae_outputs)
    reconstruction_loss = keras.backend.sum(reconstruction_loss, axis=1)

    kl_loss = -0.5 * keras.backend.sum(1 + z_log_var -
                                       keras.backend.square(z_mean) -
                                       keras.backend.exp(z_log_var), axis=1)

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
