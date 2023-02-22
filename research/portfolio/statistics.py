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
