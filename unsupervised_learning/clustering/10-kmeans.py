#!/usr/bin/env python3
"""This performs K-means on a data set with sklearn"""

import sklearn.cluster


def kmeans(X, k):
    """Here have some documentation"""
    # if not isinstance(X, np.ndarray) or len(X.shape) != 2:
    #     return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)

    kmeans_model.fit(X)

    C = kmeans_model.cluster_centers_

    clss = kmeans_model.labels_

    return C, clss