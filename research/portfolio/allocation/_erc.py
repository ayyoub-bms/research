import numpy as np
from scipy.optimize import minimize


def risk_parity(cov, leverage=1, long_only=True):

    def objective(x): return _objective(x, cov)

    options = {'ftol': 1e-10, 'maxiter': 5000}

    if leverage == 1:
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    else:
        constraints = {
            'type': 'ineq', 'fun': lambda x: np.linalg.norm(x, 1) - leverage
        }

    params = dict(method='SLSQP',
                  options=options,
                  constraints=constraints)

    if long_only:
        params.update(bounds=[(0, 1) for i in range(cov.shape[0])])

    x0 = inverse_vol(cov)

    res = minimize(objective, x0, **params)

    if res.success:
        return res.x
    else:
        raise ValueError(f'Algorithm failed to converge with reason \n {res}')


def inverse_vol(cov):
    vols = np.sqrt(np.diag(cov))
    w = 1 / vols
    return w / w.sum()


def _objective(x, cov):
    rctrb = (x.T @ cov) * x
    return np.sum(np.square(np.ravel(rctrb - rctrb.T)))
