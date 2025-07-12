#!/usr/bin/env python3
"""
Function that performs forward propagation for a bidirectional RNN
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Args:
        bi_cell: instance of BidirectionalCell used for forward propagation
        X (np.ndarray): input data for the RNN, shape (t, m, i)
                        t: number of time steps
                        m: batch size
                        i: dimensionality of the data
        h_0 (np.ndarray): initial hidden state for the forward direction, shape (m, h)
        h_t (np.ndarray): initial hidden state for the backward direction, shape (m, h)

    Returns:
        H (np.ndarray): concatenated hidden states, shape (t, m, 2 * h)
        Y (np.ndarray): outputs for each time step, shape (t, m, o)
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    # Forward hidden states
    H_f = np.zeros((t, m, h))
    h_prev_f = h_0
    for step in range(t):
        h_prev_f = bi_cell.forward(h_prev_f, X[step])
        H_f[step] = h_prev_f

    # Backward hidden states
    H_b = np.zeros((t, m, h))
    h_prev_b = h_t
    for step in reversed(range(t)):
        h_prev_b = bi_cell.backward(h_prev_b, X[step])
        H_b[step] = h_prev_b

    # Concatenate hidden states
    H = np.concatenate((H_f, H_b), axis=2)

    # Compute outputs
    Y = bi_cell.output(H)

    return H, Y
