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
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.X_s = X_s
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement (EI)

        Returns:
        - X_next: (1,) numpy.ndarray of the next best sample point
        - EI: (ac_samples,) numpy.ndarray of expected improvements
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = np.sqrt(sigma)

        if self.minimize:
            Y_opt = np.min(self.gp.Y)
            imp = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            imp = mu - Y_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(mu)
            Z[sigma > 0] = imp[sigma > 0] / sigma[sigma > 0]

        # Standard normal PDF
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z**2)

        # Standard normal CDF using erf
        cdf = 0.5 * (1 + np.erf(Z / np.sqrt(2)))

        EI = imp * cdf + sigma * pdf
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)].reshape(1)

        return X_next, EI
