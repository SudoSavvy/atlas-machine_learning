#!/usr/bin/env python3
"""Bayesian Optimization module for a noiseless 1D Gaussian Process"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Class constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Calculates the next best sample location using Expected Improvement"""
        mu, sigma = self.gp.predict(self.X_s)
        sigma = np.sqrt(sigma)

        if self.minimize:
            opt_value = np.min(self.gp.Y)
            improvement = opt_value - mu - self.xsi
        else:
            opt_value = np.max(self.gp.Y)
            improvement = mu - opt_value - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(mu)
            mask = sigma > 0
            Z[mask] = improvement[mask] / sigma[mask]

        # Standard normal PDF
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z**2)

        # Standard normal CDF using np.erf
        cdf = 0.5 * (1 + np.erf(Z / np.sqrt(2)))

        # EI formula
        EI = improvement * cdf + sigma * pdf
        EI[sigma == 0.0] = 0.0

        # Find next best point
        X_next = self.X_s[np.argmax(EI)].reshape(1)

        return X_next, EI
