#!/usr/bin/env python3
"""
rotate_image.py

This module defines a function to rotate a 3D image tensor by 90 degrees
 counter-clockwise using TensorFlow.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates a 3D image tensor by 90 degrees counter-clockwise.

    Args:
        image (tf.Tensor): A 3D TensorFlow tensor representing an image
                           with shape (height, width, channels).

    Returns:
        tf.Tensor: The image tensor rotated 90 degrees counter-clockwise.
    """
    return tf.image.rot90(image, k=1)
