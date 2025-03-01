#!/usr/bin/env python3
import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras without using Sequential.

    Parameters:
    - nx: Number of input features
    - layers: List containing the number of nodes in each layer
    - activations: List containing activation functions for each layer
    - lambtha: L2 regularization parameter
    - keep_prob: Probability of keeping a node during dropout

    Returns:
    - Keras model
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    
    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        
        if i < len(layers) - 1:  # Dropout is not applied to the last layer
            x = K.layers.Dropout(1 - keep_prob)(x)
    
    model = K.Model(inputs=inputs, outputs=x)
    return model
