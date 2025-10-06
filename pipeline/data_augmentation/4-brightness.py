#!/usr/bin/env python3
"""
change_brightness.py

This module defines a function to randomly adjust the brightness of a 3D image tensor using TensorFlow.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly adjusts the brightness of a 3D image tensor.

    Args:
        image (tf.Tensor): A 3D TensorFlow tensor representing an image
                           with shape (height, width, channels).
        max_delta (float): Maximum amount to adjust brightness. Positive values
                           increase brightness, negative values decrease it.

    Returns:
        tf.Tensor: The brightness-adjusted image tensor.
    """
    return tf.image.random_brightness(image, max_delta)
