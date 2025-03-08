#!/usr/bin/env python3

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a convolutional layer of a neural network.

    Parameters:
    - dZ (numpy.ndarray): Partial derivatives with respect to the unactivated output (m, h_new, w_new, c_new)
    - A_prev (numpy.ndarray): Output of the previous layer (m, h_prev, w_prev, c_prev)
    - W (numpy.ndarray): Kernels for the convolution (kh, kw, c_prev, c_new)
    - b (numpy.ndarray): Biases applied to the convolution (1, 1, 1, c_new)
    - padding (str): Type of padding ('same' or 'valid')
    - stride (tuple): Strides for the convolution (sh, sw)

    Returns:
    - dA_prev (numpy.ndarray): Partial derivatives with respect to the previous layer
    - dW (numpy.ndarray): Partial derivatives with respect to the kernels
    - db (numpy.ndarray): Partial derivatives with respect to the biases
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, _ = dZ.shape

    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_new) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_new) // 2
    else:
        pad_h, pad_w = 0, 0

    A_prev_padded = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            h_start, h_end = i * sh, i * sh + kh
            w_start, w_end = j * sw, j * sw + kw

            for n in range(m):
                a_slice = A_prev_padded[n, h_start:h_end, w_start:w_end, :]
                for c in range(c_new):
                    dW[:, :, :, c] += a_slice * dZ[n, i, j, c]
                    dA_prev_padded[n, h_start:h_end, w_start:w_end, :] += W[:, :, :, c] * dZ[n, i, j, c]

    if padding == "same":
        dA_prev = dA_prev_padded[:, pad_h:-pad_h, pad_w:-pad_w, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
