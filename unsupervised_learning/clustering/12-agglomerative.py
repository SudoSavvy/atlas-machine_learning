#!/usr/bin/env python3
"""
Performs agglomerative clustering with Ward linkage and plots dendrogram.
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset.

    Parameters:
    - X: np.ndarray of shape (n, d) containing the dataset
    - dist: float, maximum cophenetic distance for all clusters

    Returns:
    - clss: np.ndarray of shape (n,) with cluster indices for each point
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(linkage, t=dist, criterion='distance')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return clss
