#!/usr/bin/env python3
"""Bayesian Optimization module for a noiseless 1D Gaussian Process"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        Parameters:
        - f: the black-box function to be optimized
        - X_init: (t, 1) numpy.ndarray of initial input samples
        - Y_init: (t, 1) numpy.ndarray of initial output samples
        - bounds: tuple of (min, max) bounds of the optimization space
        - ac_samples: number of acquisition sample points
        - l: length parameter for the kernel
        - sigma_f: output standard deviation of the black-box function
        - xsi: exploration-exploitation factor
        - minimize: boolean for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

        X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

        self.X_s = X_s
        self.xsi = xsi
        self.minimize = minimize
