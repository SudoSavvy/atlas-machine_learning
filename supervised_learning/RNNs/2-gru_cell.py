#!/usr/bin/env python3
"""
Class that represents a GRU (Gated Recurrent Unit) cell
"""

import numpy as np


class GRUCell:
    """
    Represents a single GRU cell performing one step of
    forward propagation in a GRU-based RNN.
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): dimensionality of the data input
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs

        Public instance attributes:
            Wz, bz: weights and bias for update gate (shape: (i + h, h), (1, h))
            Wr, br: weights and bias for reset gate  (shape: (i + h, h), (1, h))
            Wh, bh: weights and bias for candidate hidden state (shape: (i + h, h), (1, h))
            Wy, by: weights and bias for output (shape: (h, o), (1, o))
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev (np.ndarray): previous hidden state, shape (m, h)
            x_t (np.ndarray): input data at time t, shape (m, i)

        Returns:
            h_next (np.ndarray): next hidden state, shape (m, h)
            y (np.ndarray): output of the cell, shape (m, o)
        """
        m, h = h_prev.shape

        # Concatenate x_t and h_prev
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate z_t
        z_t = self.sigmoid(np.matmul(concat, self.Wz) + self.bz)

        # Reset gate r_t
        r_t = self.sigmoid(np.matmul(concat, self.Wr) + self.br)

        # Candidate hidden state
        r_h_prev = r_t * h_prev
        concat_candidate = np.concatenate((r_h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(concat_candidate, self.Wh) + self.bh)

        # Final hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_hat

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
