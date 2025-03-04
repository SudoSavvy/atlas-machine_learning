#!/usr/bin/env python3
"""
Function to perform a convolution on images using multiple kernels.
"""
import numpy as np

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Args:
        images (numpy.ndarray): Shape (m, h, w, c), multiple images.
            - m: Number of images.
            - h: Height of the images.
            - w: Width of the images.
            - c: Number of channels.
        kernels (numpy.ndarray): Shape (kh, kw, c, nc), the kernels for convolution.
            - kh: Height of a kernel.
            - kw: Width of a kernel.
            - c: Number of channels.
            - nc: Number of kernels.
        padding (str or tuple): Either 'same', 'valid', or a tuple (ph, pw).
            - 'same': Performs same convolution (zero-padding to keep size).
            - 'valid': Performs valid convolution (no padding).
            - (ph, pw): Custom padding for height and width.
        stride (tuple): (sh, sw), the stride for height and width.
            - sh: Stride for the height of the image.
            - sw: Stride for the width of the image.

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1
    convolved = np.zeros((m, new_h, new_w, nc))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(nc):
                convolved[:, i, j, k] = np.sum(
                    padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved
