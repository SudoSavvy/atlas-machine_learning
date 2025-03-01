#!/usr/bin/env python3
import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with Keras.

    Parameters:
    - nx: Number of input features
    - layers: List containing the number of nodes in each layer
    - activations: List containing activation functions for each layer
    - lambtha: L2 regularization parameter
    - keep_prob: Probability of keeping a node during dropout

    Returns:
    - Keras model
    """
    model = K.Sequential()
    
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)
            ))
        
        if i < len(layers) - 1:  # Dropout is not applied to the last layer
            model.add(K.layers.Dropout(1 - keep_prob))
    
    return model
