#!/usr/bin/env python3
"""
flip_image.py

This module provides a function to flip a 3D image tensor horizontally using TensorFlow.

Requirements:
- Only TensorFlow may be imported.
- Must pass pycodestyle validation.
- Output must match the main file example.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flip a 3D image tensor horizontally.

    Args:
        image (tf.Tensor): A 3D tensor representing an image (height, width, channels).

    Returns:
        tf.Tensor: The horizontally flipped image tensor.
    """
    return tf.image.flip_left_right(image)
