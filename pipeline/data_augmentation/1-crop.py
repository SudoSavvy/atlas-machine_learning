#!/usr/bin/env python3
"""
crop_image.py

This module defines a function to perform a random crop on a 3D image
 tensor using TensorFlow.
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop on a 3D image tensor.

    Args:
        image (tf.Tensor): A 3D TensorFlow tensor representing an image
                           with shape (height, width, channels).
        size (tuple): A tuple (crop_height, crop_width, channels) specifying
                      the size of the crop.

    Returns:
        tf.Tensor: A randomly cropped image tensor of the specified size.
    """
    return tf.image.random_crop(image, size)
