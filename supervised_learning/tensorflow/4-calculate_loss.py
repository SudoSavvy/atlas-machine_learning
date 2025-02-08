#!/usr/bin/env python3

"""
Module for calculating accuracy and loss of predictions using TensorFlow v1.
"""

import tensorflow.compat.v1 as tf

def calculate_accuracy(y, y_pred):
    """
    Compute the accuracy of a prediction.

    Args:
        y (tf.Tensor): Placeholder tensor containing the true labels.
        y_pred (tf.Tensor): Tensor containing the network’s predictions.

    Returns:
        tf.Tensor: A tensor representing the decimal accuracy of the prediction.
    """
    correct_predictions = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def calculate_loss(y, y_pred):
    """
    Compute the softmax cross-entropy loss of a prediction.

    Args:
        y (tf.Tensor): Placeholder tensor containing the true labels.
        y_pred (tf.Tensor): Tensor containing the network’s predictions.

    Returns:
        tf.Tensor: A tensor representing the loss of the prediction.
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))
    return loss
