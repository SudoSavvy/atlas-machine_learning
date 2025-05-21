#!/usr/bin/env python3
"""documentation"""

import sklearn.mixture


def gmm(X, k):
    """documentation"""

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None

    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)
    gmm_model.fit(X)

    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic