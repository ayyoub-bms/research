from ._herc import hierarchical_equal_risk_contribution
from ._erc import risk_parity, inverse_vol
from ._mdp import maximum_diversification
from ._mv import minimum_volatility
from ._meanv import (
    mean_variance_with_risk_aversion,
    mean_variance_minimize_risk,
    mean_variance_maximize_returns,
    mean_variance
)


def normalize_weights(w):
    return w / w.sum()
