#!/usr/bin/env python3
"""
Function to perform pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images.

    Args:
        images (numpy.ndarray): Shape (m, h, w, c), multiple images.
            - m: Number of images.
            - h: Height of the images.
            - w: Width of the images.
            - c: Number of channels.
        kernel_shape (tuple): (kh, kw), the kernel shape for pooling.
            - kh: Height of the kernel.
            - kw: Width of the kernel.
        stride (tuple): (sh, sw), the stride for height and width.
            - sh: Stride for the height of the image.
            - sw: Stride for the width of the image.
        mode (str): Type of pooling ('max' for max pooling, 'avg'
        for average pooling).

    Returns:
        numpy.ndarray: Pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1
    pooled = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            if mode == 'max':
                pooled[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2)
                )

    return pooled
