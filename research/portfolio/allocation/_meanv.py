import numpy as np
import cvxpy as cp

from ._constraints import _add_constraints


def mean_variance(inv_cov, mean_returns, minimum_return, long_only):
    nb_stocks = len(mean_returns)
    w_mv = minimum_volatility(inv_cov, long_only=long_only)
    mv_mean_return = w_mv @ mean_returns

    if mv_mean_return >= minimum_return:
        return w_mv

    one = np.ones(nb_stocks)
    sm = inv_cov @ mean_returns
    se = inv_cov @ one

    me = mean_returns.T @ se
    mm = mean_returns.T @ sm
    ee = one.T @ inv_cov @ one
    denom = mm * ee - me*me

    if denom > 0:
        theta = (minimum_return * me * ee - me * me) / denom
        w_mkt = sm / sm.sum()
        return theta * w_mkt + (1 - theta) * w_mv
    else:
        raise ValueError('Problem infeasible')


def mean_variance_with_risk_aversion(cov, mean_returns, aversion,
                                     leverage=1, long_only=True):
    n = len(mean_returns)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    gamma.value = aversion
    ptf_ret = portfolio_returns(mean_returns, w)
    ptf_var = cp.quad_form(w, cov)
    constraints = []

    if long_only:
        constraints.append(w >= 0)

    if leverage == 1:
        constraints.append(cp.sum(w) == 1)

    elif leverage > 1:
        constraints.append(cp.norm(w, 1) <= leverage)
    else:
        raise ValueError('Leverage should be >= 1')

    prob = cp.Problem(cp.Maximize(ptf_ret - gamma * ptf_var), constraints)
    prob.solve()
    return w.value


def mean_variance_maximize_returns(cov, mean_returns, maximum_variance,
                                   leverage=1, long_only=True):
    n = len(mean_returns)
    w = cp.Variable(n)

    ptf_ret = portfolio_returns(mean_returns, w)
    ptf_var = cp.quad_form(w, cov)

    constraints = _add_constraints(
        w,
        ptf_var <= maximum_variance,  # Add variance upper bound
        leverage=leverage,
        long_only=long_only
    )

    prob = cp.Problem(cp.Maximize(ptf_ret), constraints)
    prob.solve()
    return w.value


def mean_variance_minimize_risk(cov, mean_returns, minimum_return,
                                leverage=1, long_only=True):
    n = len(mean_returns)
    w = cp.Variable(n)

    ptf_ret = portfolio_returns(mean_returns, w)
    ptf_var = cp.quad_form(w, cov)

    constraints = _add_constraints(
        w,
        ptf_ret >= minimum_return,  # Add returns lower bound
        leverage=leverage,
        long_only=long_only
    )

    prob = cp.Problem(cp.Minimize(ptf_var), constraints)
    prob.solve()
    return w.value