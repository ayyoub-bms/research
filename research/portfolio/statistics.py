import numpy as np


def portfolio_variance(cov: np.ndarray, w: np.ndarray) -> float:
    return w.T @ cov @ w


def portfolio_volatility(cov: np.ndarray, w: np.ndarray) -> float:
    return np.sqrt(portfolio_variance(cov, w))


def marginal_risk_contrib(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ Returns the sensitivity of portfolio risk to small changes in weights.
    """
    return (cov @ w) / portfolio_volatility(cov, w)


def total_risk_contribution(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ Marginal risk contribution of each stock times weights"""
    return w * marginal_risk_contrib(cov, w)


def risk_budget(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """ Returns the % risk budget of each stock"""
    return total_risk_contribution(cov, w) / portfolio_volatility(cov, w)


def max_drawdown(x):
    return np.minimum.accumulate(drawdown(x)).min() * 100


def drawdown(x):
    x = x[~np.isnan(x)]
    rollingmax = np.maximum.accumulate(x)
    drawdown = (x - rollingmax) / rollingmax
    return drawdown


def gini(x):
    x = x[~np.isnan(x)]
    n = len(x)
    w = sorted(x)
    z = 2*(np.arange(n) + 1) - n - 1
    return np.sum(z * w) / (n*np.sum(w))


def herfindahl(x):
    x = x[~np.isnan(x)]
    n = 1/len(x)
    ht = np.power(x, 2).sum()
    return (ht - n) / (1-n)


def turnover(weights, previous_weights, axis=1):
    return .5 * abs((weights - previous_weights)).sum(axis=axis)


def diversification_ratio(cov, w):
    vol = np.sqrt(np.diag(cov))
    return (vol.T @ w) / portfolio_volatility(cov, w)


def average_correlation(corr):
    n = corr.shape[0]
    return (np.sum(corr) - n) / (2*n)
