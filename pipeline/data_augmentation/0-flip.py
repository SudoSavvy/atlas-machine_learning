#!/usr/bin/env python3
"""
flip_image.py

This module defines a function to flip a 3D image tensor horizontally using
 TensorFlow.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips a 3D image tensor horizontally.

    Args:
        image (tf.Tensor): A 3D TensorFlow tensor representing an image
                           with shape (height, width, channels).

    Returns:
        tf.Tensor: A horizontally flipped image tensor.
    """
    return tf.image.flip_left_right(image)
