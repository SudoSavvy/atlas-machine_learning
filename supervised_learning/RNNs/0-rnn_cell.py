#!/usr/bin/env python3
"""
Class that represents a cell of a simple RNN
"""

import numpy as np


class RNNCell:
    """
    Represents a single RNN cell performing one step of
    forward propagation in a simple RNN.
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): dimensionality of the data input
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs

        Public instance attributes:
            Wh (np.ndarray): weights for the concatenated hidden state and input data
                              shape (h + i, h)
            bh (np.ndarray): biases for the hidden state
                              shape (1, h)
            Wy (np.ndarray): weights for the output
                              shape (h, o)
            by (np.ndarray): biases for the output
                              shape (1, o)
        """
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev (np.ndarray): previous hidden state, shape (m, h)
            x_t (np.ndarray): data input at time t, shape (m, i)

        Returns:
            h_next (np.ndarray): next hidden state, shape (m, h)
            y (np.ndarray): output of the cell, shape (m, o)
        """
        # Concatenate previous hidden state and current input along features
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state calculation with tanh activation
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Raw output (logits)
        y_linear = np.matmul(h_next, self.Wy) + self.by

        # Softmax activation for output
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y
