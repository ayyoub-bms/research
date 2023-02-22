import numpy as np


from collections import deque
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

from ._erc import inverse_vol
from ..statistics import total_risk_contribution, portfolio_volatility


def hierarchical_equal_risk_contribution(cov, pmin=5, pmax=15):
    """Computes the HERC allocations

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
    cluster_risk_contrib = dict()
    cluster_weights = dict()
    link = np.asarray(link, dtype=int)
    leaves, clusters = _build_clusters(link, nb_clusters)

    for i, c in clusters.items():
        V = cov.iloc[c, c].values
        w = inverse_vol(np.sqrt(np.diag(V)))
        cluster_risk_contrib[i] = total_risk_contribution(V, w).sum()
        cluster_weights[i] = w

    for i in range(nb_clusters-1):
        left = link[nb_stocks - i - 2, 0]
        right = link[nb_stocks - i - 2, 1]
        lstocks = clusters[left]
        rstocks = clusters[right]
        rcl = cluster_risk_contrib[left]
        rcr = cluster_risk_contrib[right]
        rc = rcl + rcr
        a = rcr / rc
        allocations[lstocks] *= a
        allocations[rstocks] *= (1 - a)

    for leaf in leaves:
        stocks = clusters[leaf]
        allocations[stocks] *= cluster_weights[leaf]
    return allocations


def _get_number_of_clusters(link, diss, pmin=2, pmax=10):
    scores = deque()
    for i in range(pmin, pmax+1):
        classes = fcluster(link, i, criterion='maxclust')
        scores.append(silhouette_score(diss, classes))
    return np.argmax(scores) + pmin


def _get_leaves(link, p):
    n = link[-1, 3]
    data = deque()
    for i in range(p-1):
        data.append(link[n - i - 2, 0])
        data.append(link[n - i - 2, 1])
    return sorted(data)[:p]


def _get_stocks(link, cluster_id):
    """Return the leaves of a given node in the tree
    i.e returns the stocks that have been merged to form
    the cluster `cluster_id`
    """
    n = link[-1, 3]
    if cluster_id < n:
        return [cluster_id]

    ids = set()
    not_visited = deque()
    not_visited.append(cluster_id)
    size = 1
    while not_visited:
        x = not_visited.popleft()
        x = x - n
        left = link[x, 0]
        right = link[x, 1]
        if left >= n:
            not_visited.append(left)
        else:
            ids.add(left)
        if right >= n:
            not_visited.append(right)
        else:
            ids.add(right)
    return list(ids)


def _build_clusters(link, nb_clusters):
    """ For each node in the cut tree, get the stocks belonging
    to the node
    """
    def _clusters2stocks(root):
        """ Attribute to each cluster the stocks belonging to it

        Recursively add the appropriate stocks from the leaf nodes of the tree
        to each parent. The idea is to merge the leafs' stocks
        for each parent up to the root node by concatenating the stocks
        for the left and right childs. If the clusters dict contains the node, nothing
        needs to be done, otherwise either we are in a node that is a child of
        one of the leaf nodes hence we return an empty list or we are one of the parents
        and we use recursion to concatenate the leaf's stocks.
        """
        if root < lmin:
            return []
        try:
            return clusters[root]
        except KeyError:
            clusters[root] = (
                _clusters2stocks(link[root - n, 0]) +
                _clusters2stocks(link[root - n, 1])
            )
            return clusters[root]


    clusters = dict()
    level_clusters = _get_leaves(link, nb_clusters)
    lmin = np.inf
    n = link[-1][3]
    for l in level_clusters:
        # retrieves the stocks for the node from the leafs of the tree
        clusters[l] = _get_stocks(link, l)
        lmin = min(lmin, l)

    _clusters2stocks(link[-1, 0])
    _clusters2stocks(link[-1, 1])

    return level_clusters, clusters
