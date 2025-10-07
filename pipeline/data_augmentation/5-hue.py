#!/usr/bin/env python3
import tensorflow as tf

def change_hue(image, delta):
    """
    Changes the hue of an image by a specified delta.

    Parameters:
    image (tf.Tensor): A 3D tensor representing the image to be altered.
                       The tensor should have shape (height, width, channels)
                       and values in the range [0, 1].
    delta (float): The amount by which to change the hue. Should be in the
                   range [-1.0, 1.0].

    Returns:
    tf.Tensor: A 3D tensor representing the image with adjusted hue.
    """
    return tf.image.adjust_hue(image, delta)
