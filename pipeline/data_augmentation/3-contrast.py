#!/usr/bin/env python3
"""
change_contrast.py

This module defines a function to randomly adjust the contrast of a 3D
 image tensor using TensorFlow.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of a 3D image tensor.

    Args:
        image (tf.Tensor): A 3D TensorFlow tensor representing an image
                           with shape (height, width, channels).
        lower (float): Lower bound of the contrast adjustment factor.
        upper (float): Upper bound of the contrast adjustment factor.

    Returns:
        tf.Tensor: The contrast-adjusted image tensor.
    """
    return tf.image.random_contrast(image, lower, upper)
