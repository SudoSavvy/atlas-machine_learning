#!/usr/bin/env python3
"""
Function to perform a convolution on grayscale images with different padding and stride options.
"""
import numpy as np

def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w), multiple grayscale images.
            - m: Number of images.
            - h: Height in pixels of the images.
            - w: Width in pixels of the images.
        kernel (numpy.ndarray): Shape (kh, kw), the kernel for convolution.
            - kh: Height of the kernel.
            - kw: Width of the kernel.
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
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    
    if isinstance(padding, str):
        if padding == 'same':
            ph = max((h - 1) * sh + kh - h, 0) // 2
            pw = max((w - 1) * sw + kw - w, 0) // 2
        elif padding == 'valid':
            ph, pw = 0, 0
        else:
            raise ValueError("Invalid padding type. Use 'same', 'valid', or a tuple (ph, pw).")
    else:
        ph, pw = padding
    
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1
    convolved = np.zeros((m, new_h, new_w))
    
    for i in range(new_h):
        for j in range(new_w):
            region = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            convolved[:, i, j] = np.tensordot(region, kernel, axes=((1, 2), (0, 1)))
    
    return convolved
