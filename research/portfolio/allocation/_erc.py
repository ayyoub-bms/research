import numpy as np
from scipy.optimize import minimize


def risk_parity(cov, leverage=1, long_only=True, scale=1e-5):

    def objective(x): return _objective(x, cov, scale)

    options = {'ftol': 1e-6, 'maxiter': 1000}

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

    n = cov.shape[0]
    x0 = np.ones(n) / n

    result = minimize(objective, x0, **params)

    if result.success:
        return result.x
    else:
        raise ValueError(f'Algorithm failed to converge with reason \n {res}')


def inverse_vol(cov):
    vols = np.sqrt(np.diag(cov))
    w = 1 / vols
    return w / w.sum()


def _objective(x, cov, scale):
    rc = x.dot(cov) * x
    a = np.reshape(rc, (len(rc), 1))
    risk_diffs = a - a.transpose()
    sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
    return sum_risk_diffs_squared / scale

