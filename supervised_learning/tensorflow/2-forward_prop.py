#!/usr/bin/env python3
"""Module that creates the forward propagation graph for a neural network"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for a neural network.

    Args:
        x (tf.Tensor): Placeholder for the input data.
        layer_sizes (list): List containing the number of nodes in each layer.
        activations (list): List containing the activation functions for each
        layer.

    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """
    output = x
    for i in range(len(layer_sizes)):
        output = create_layer(output, layer_sizes[i], activations[i])
    return output
