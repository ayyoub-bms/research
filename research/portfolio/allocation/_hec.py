import numpy as np

from scipy.cluster.hierarchy import linkage
from ._erc import inverse_vol
from ..statistics import average_correlation
from ._hierarchy import (
        _build_clusters,
        _get_number_of_clusters
)


def hierarchical_equal_correlations(cov, pmin=5, pmax=15):
    """Computes the HEC allocations

    Parameters:
    -----------
    cov: The covariance matrix
    pmin: The minimum number of clusters
    pmax: The maximum number of clusters

    Returns:
    --------

        The weights for each asset.


    Note:
    -----
        The Silhouette score is used to identify the optimal number of clusters
        between `pmin` and `pmax` and the Ward hierarchical clustering algorith
        is the one used here.
    """

    nb_stocks = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    corr = cov / (vols @ vols.T)
    dissimilarity = np.sqrt(2*(1-corr))

    link = linkage(dissimilarity, method='ward', optimal_ordering=True)
    nb_clusters = _get_number_of_clusters(link, dissimilarity, pmin, pmax)

    allocations = np.ones(nb_stocks)
    cluster_avg_corr = dict()
    cluster_weights = dict()
    link = np.asarray(link, dtype=int)
    leaves, clusters = _build_clusters(link, nb_clusters)

    for i, c in clusters.items():
        V = cov.iloc[c, c].values
        C = corr.iloc[c, c].values
        w = inverse_vol(V)
        cluster_avg_corr[i] = average_correlation(C)
        cluster_weights[i] = w

    for i in range(nb_clusters-1):
        left = link[nb_stocks - i - 2, 0]
        right = link[nb_stocks - i - 2, 1]
        lstocks = clusters[left]
        rstocks = clusters[right]
        acl = cluster_avg_corr[left]
        acr = cluster_avg_corr[right]
        rc = acl + acr
        a = acr / rc
        allocations[lstocks] *= a
        allocations[rstocks] *= (1 - a)

    for leaf in leaves:
        stocks = clusters[leaf]
        allocations[stocks] *= cluster_weights[leaf]
    return allocations
