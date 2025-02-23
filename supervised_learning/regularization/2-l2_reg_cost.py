#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    
    Parameters:
    cost (tf.Tensor): Tensor containing the cost of the network without L2 regularization.
    model (tf.keras.Model): Keras model that includes layers with L2 regularization.
    
    Returns:
    tf.Tensor: A tensor containing the total cost for each layer of the network, accounting for L2 regularization.
    """
    reg_cost = cost + tf.add_n(model.losses)
    return reg_cost
