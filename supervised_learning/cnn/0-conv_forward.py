#!/usr/bin/env python3
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network.
    
    Parameters:
    - A_prev (numpy.ndarray): Input data of shape (m, h_prev, w_prev, c_prev)
    - W (numpy.ndarray): Kernels of shape (kh, kw, c_prev, c_new)
    - b (numpy.ndarray): Biases of shape (1, 1, 1, c_new)
    - activation (function): Activation function applied to the convolution
    - padding (str): Type of padding ('same' or 'valid')
    - stride (tuple): Strides (sh, sw)
    
    Returns:
    - Output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    
    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0) // 2
        pw = max((w_prev - 1) * sw + kw - w_prev, 0) // 2
    else:  # valid padding
        ph, pw = 0, 0
    
    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    
    h_new = (h_prev + 2 * ph - kh) // sh + 1
    w_new = (w_prev + 2 * pw - kw) // sw + 1
    
    Z = np.zeros((m, h_new, w_new, c_new))
    
    for i in range(h_new):
        for j in range(w_new):
            h_start, h_end = i * sh, i * sh + kh
            w_start, w_end = j * sw, j * sw + kw
            
            slice_A = A_prev_padded[:, h_start:h_end, w_start:w_end, :]
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(slice_A * W[..., k], axis=(1, 2, 3)) + b[..., k]
    
    return activation(Z)
