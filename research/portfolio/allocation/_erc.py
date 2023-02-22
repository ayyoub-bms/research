import numpy as np


def risk_parity(cov, leverage=1, long_only=True):
    nb_stocks, _ = cov.shape

    w = cp.Variable(nb_stocks)
    trc = np.reshape((cov @ w) * w, (nb_stocks, 1))
    mat = trc - trc.T
    objective = np.sum(np.square(mat))
    constraints = _add_constraints(leverage=leverage, long_only=long_only)
    prob = cp.Problem(cp.Minimize(objective), constraints=constraints)
    return w.value


def inverse_vol(vols):
    w = 1 / vols
    return w / w.sum()
