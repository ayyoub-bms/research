import numpy as np
from collections import deque
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score


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
        to each parent. The idea is to merge the leafs' stocks for each parent
        up to the root node by concatenating the stocks for the left and right
        childs. If the clusters dict contains the node, nothing needs to be
        done, otherwise either we are in a node that is a child of one of the
        leaf nodes hence we return an empty list or we are one of the parents
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
    n = link[-1, 3]
    for i in level_clusters:
        # retrieves the stocks for the node from the leafs of the tree
        clusters[i] = _get_stocks(link, i)
        lmin = min(lmin, i)

    _clusters2stocks(link[-1, 0])
    _clusters2stocks(link[-1, 1])

    return level_clusters, clusters
