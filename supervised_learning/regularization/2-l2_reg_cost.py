#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    
    Parameters:
    cost (tf.Tensor): Tensor containing the cost of the network without L2 regularization.
    model (tf.keras.Model): Keras model that includes layers with L2 regularization.
    
    Returns:
    tf.Tensor: Total cost of the network, including L2 regularization.
    """
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables if 'kernel' in var.name])
    reg_cost = cost + tf.reduce_sum(model.losses)
    return reg_cost
