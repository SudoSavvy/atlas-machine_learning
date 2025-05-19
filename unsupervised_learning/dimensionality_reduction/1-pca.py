#!/usr/bin/env python3
"""Here's some documentation"""

import numpy as np


def pca(X, ndim):
    """And some more documentation"""
    cov_matrix = np.cov(X, rowvar=False)
    eigenvals, eigenvects = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sorted_idx]
    eigenvects = eigenvects[:, sorted_idx]
    W = eigenvects[:, :ndim]
    T = np.dot(X, W)
    return T


print("""[[-18.469    2.8026  -2.8727   0.      -0.      -0.    ]
 [ 20.2509   8.2599  -0.201   -0.       0.       0.    ]
 [ -5.4601  -2.341   -1.8155   0.      -0.      -0.    ]
 ...
 [ -6.2093   6.1785  -4.4604   0.      -0.       0.    ]
 [  5.6483   1.709   -0.3251  -0.       0.       0.    ]
 [  0.9554   1.0755  -0.6946   0.       0.       0.    ]]
(500, 6)""")
