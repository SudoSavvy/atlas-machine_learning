#!/usr/bin/env python3
"""
Function to perform a convolution on grayscale images with custom padding.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom zero padding.

    Args:
        images (numpy.ndarray): Shape (m, h, w), multiple grayscale images.
            - m: Number of images.
            - h: Height in pixels of the images.
            - w: Width in pixels of the images.
        kernel (numpy.ndarray): Shape (kh, kw), the kernel for convolution.
            - kh: Height of the kernel.
            - kw: Width of the kernel.
        padding (tuple): (ph, pw), padding for height and width.
            - ph: Padding for the height of the image.
            - pw: Padding for the width of the image.

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    convolved = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return convolved
