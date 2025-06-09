#!/usr/bin/env python3
"""creates the class for a
Bayesian Optimization"""

import numpy as np
from scipy.stats import norm
GP = __import__("2-gp").GaussianProcess


class BayesianOptimization:
    """Beysianbby"""

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True,
    ):

        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.l = l
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Documentation"""
        mu_s, sigma_s = self.gp.predict(self.X_s)

        sigma_s = np.maximum(sigma_s, 1e-9)

        if self.minimize:
            f_best = np.min(self.gp.Y)
            imp = (f_best - mu_s) - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            imp = (mu_s - y) - self.xsi

        with np.errstate(divide="warn"):
            Z = imp / sigma_s
            ei = imp * norm.cdf(Z) + sigma_s * norm.pdf(Z)

        ei[sigma_s == 0.0] = 0.0

        max_ei_idx = np.argmax(ei)
        X_next = self.X_s[max_ei_idx]

        return X_next, ei

    def optimize(self, iterations=100):
        """Optomatron"""

        for i in range(iterations):
            X_next, _ = self.acquisition()

            if np.any(np.isclose(X_next, self.gp.X)):
                break

            Y_next = self.f(X_next)

            self.gp.X = np.vstack([self.gp.X, X_next.reshape(-1, 1)])
            self.gp.Y = np.vstack([self.gp.Y, Y_next.reshape(-1, 1)])

            self.gp.K = self.gp.kernel(self.gp.X, self.gp.X)

        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt