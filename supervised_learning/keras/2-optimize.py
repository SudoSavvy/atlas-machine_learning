#!/usr/bin/env python3
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with categorical crossentropy loss and accuracy metrics.

    Parameters:
    - network: The model to optimize
    - alpha: Learning rate
    - beta1: First Adam optimization parameter
    - beta2: Second Adam optimization parameter

    Returns:
    - None
    """
    optimizer = K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
