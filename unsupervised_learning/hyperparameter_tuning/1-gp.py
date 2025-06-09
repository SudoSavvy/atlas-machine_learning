#!/usr/bin/env python3
"""GaussianProcess module for 1D noiseless Gaussian process with prediction"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor

        Parameters:
        - X_init: (t, 1) numpy.ndarray of initial inputs
        - Y_init: (t, 1) numpy.ndarray of initial outputs
        - l: length-scale of the kernel
        - sigma_f: standard deviation of the function output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF) kernel

        Parameters:
        - X1: (m, 1) numpy.ndarray
        - X2: (n, 1) numpy.ndarray

        Returns:
        - Covariance matrix as numpy.ndarray of shape (m, n)
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) \
            + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a GP

        Parameters:
        - X_s: (s, 1) numpy.ndarray of sample locations

        Returns:
        - mu: (s,) numpy.ndarray of means
        - sigma: (s,) numpy.ndarray of variances
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # Mean vector
        mu_s = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        # Covariance matrix
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma_s = np.diag(cov_s)

        return mu_s, sigma_s
