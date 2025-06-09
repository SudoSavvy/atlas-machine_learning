#!/usr/bin/env python3
"""Bayesian Optimization"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.bounds = bounds

    def acquisition(self):
        """
        Computes the next best sample location using Expected Improvement
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu.reshape(-1, 1) - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu.reshape(-1, 1) - best - self.xsi

        Z = np.zeros_like(imp)
        std_nonzero = sigma != 0
        Z[std_nonzero] = imp[std_nonzero] / sigma[std_nonzero]

        # Use numerical approximation of normal PDF and CDF
        cdf = 0.5 * (1 + np.erf(Z / np.sqrt(2)))
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * Z**2)

        EI = imp * cdf + sigma * pdf
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next.reshape(1,), EI.flatten()

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(X_next, self.gp.X)):
                break  # Stop if X_next was already sampled

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
