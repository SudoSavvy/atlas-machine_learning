#!/usr/bin/env python3
"""
Function to perform a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images with
    zero padding if necessary.

    Args:
        images (numpy.ndarray): Shape (m, h, w), multiple
        grayscale images.
            - m: Number of images.
            - h: Height in pixels of the images.
            - w: Width in pixels of the images.
        kernel (numpy.ndarray): Shape (kh, kw), the kernel
        for convolution.
            - kh: Height of the kernel.
            - kw: Width of the kernel.

    Returns:
        numpy.ndarray: Convolved images with same spatial
        dimensions as the input.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    convolved = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            convolved[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return convolved
