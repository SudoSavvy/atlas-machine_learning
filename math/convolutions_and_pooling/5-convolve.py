#!/usr/bin/env python3
"""Convolve(images, kernels, padding='same', stride=(1, 1)): Module
-- Multiple Kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images using multiple
    kernels:

    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
    images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the
    kernels for the convolution
    kh is the height of a kernel
    kw is the width of a kernel
    nc is the number of kernels
    padding is either a tuple of (ph, pw), 'same', or 'valid'
    if 'same', performs a same convolution
    if 'valid', performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0's
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    You are only allowed to use three for loops; any other loops of any kind
    are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    # unpack images, kernels, stride
    (m, h, w, c), (kh, kw, kc, nc) = images.shape, kernels.shape
    (sh, sw) = stride

    # statements to determine padding
    if padding == 'same':
        # if same then calculate the padding size
        pad_h = (((h - 1) * sh) + kh - h) // 2 + 1
        pad_w = (((w - 1) * sw) + kw - w) // 2 + 1
    elif padding == 'valid':
        # if valid then no padding for height and width
        pad_h = 0
        pad_w = 0
    else:
        # if padding is a tuple then use padding values
        pad_h, pad_w = padding

    # pad the images with zeros with the padding values
    padded_images = np.pad(
        images,
        ((0, 0),
         (pad_h, pad_h),
         (pad_w, pad_w),
         (0, 0)),
    )

    # calculate output height and width
    output_h = (h + 2 * pad_h - kh) // sh + 1
    output_w = (w + 2 * pad_w - kw) // sw + 1

    # init the outcome
    convolved_images = np.zeros((m, output_h, output_w, nc))

    # apply convolution
    for k in range(nc):
        for y in range(output_h):
            for x in range(output_w):
                # retrieve the current stride region
                current_stride = padded_images[
                    :,
                    y * sh:y * sh + kh,
                    x * sw:x * sw + kw,
                    :
                ]

                # apply tensordot function with kernels
                convolved_images[:, y, x, k] = np.tensordot(
                    current_stride, kernels[:, :, :, k],
                    axes=((1, 2, 3), (0, 1, 2))
                )

    # return a numpy.ndarray containing the convolved images
    return (convolved_images)
