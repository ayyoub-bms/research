import numpy as np
import cvxpy as cp

from ._constraints import _add_constraints


def minimum_volatility(cov, leverage=1, long_only=True):
    """ Returns the weights for a minimum volatility portfolio

    Parameters:
    -----------
        cov: numpy ndarray
            The inverse of the covariance matrix of stock returns
        leverage: float
            The upper bound for the sum of weights if leverage > 1 or
            the weights must sum to 1 for a leverage of 1.
        long_only: bool
            The weights positivity constraint (long only portfolios)

    Returns:
    --------
        w: numpy array
            The portfolio allocations
    """
    n = cov.shape[0]

    if long_only:
        w = cp.Variable(n)
        ptf_var = cp.quad_form(w, cov)
        constraints = _add_constraints(
            w,
            leverage=leverage,
            long_only=True
        )
        prob = cp.Problem(cp.Minimize(ptf_var), constraints)
        prob.solve()
        return w.value

    one = np.ones(n)
    w = np.linalg.pinv(cov) @ one
    w = w / w.sum()
    return w
