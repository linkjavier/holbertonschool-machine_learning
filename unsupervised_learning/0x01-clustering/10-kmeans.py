#!/usr/bin/env python3
""" Hello, sklearn! """

import sklearn.cluster


def kmeans(X, k):
    """ Function that performs K-means on a dataset """

    Kmean = sklearn.cluster.KMeans(n_clusters=k)
    Kmean.fit(X)

    C = Kmean.cluster_centers_
    clss = Kmean.labels_

    return C, clss
