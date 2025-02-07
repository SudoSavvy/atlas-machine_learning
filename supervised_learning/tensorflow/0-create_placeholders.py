#!/usr/bin/env python3
"""Module that creates placeholders a neural network using TensorFlow v1"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates and returns TensorFlow placeholders for input data and
    labels.

    Args:
        nx (int): Number of feature columns in the data.
        classes (int): Number of classes in the classifier.

    Returns:
        tuple: A pair of placeholders (x, y).
            - x: Placeholder for input data, shape (None, nx), dtype
            float32
            - y: Placeholder for one-hot labels, shape (None, classes),
            dtype float32
    """
    x = tf.placeholder(dtype=tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(dtype=tf.float32, shape=(None, classes), name="y")
    return x, y
