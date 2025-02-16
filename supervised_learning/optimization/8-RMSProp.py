#!/usr/bin/env python3

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the RMSProp optimization operation in TensorFlow.

    Parameters:
    alpha (float): Learning rate.
    beta2 (float): RMSProp weight (Discounting factor).
    epsilon (float): Small number to avoid division by zero.

    Returns:
    tf.keras.optimizers.RMSprop: RMSProp optimizer instance.
    """
    return tf.keras.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)
