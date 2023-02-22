import numpy as np


def minimum_volatility(inv_cov, long_only=True):
    """Returns the weights for a minimum volatility portfolio

    Parameters:
    -----------
        inv_cov: numpy ndarray
            The inverse of the covariance matrix of stock returns
        long_only: bool
            The weights positivity constraint (long only portfolios)

    Returns:
    --------
        w: numpy array
            The portfolio allocations
    """
    n = inv_cov.shape[0]
    one = np.ones(n)
    w = inv_cov @ one
    w = w / w.sum()
    if long_only:
        w = .5 * (w + 1/n)
    return w
