import cvxpy as cp


def _add_constraints(weights, *args, leverage=1, long_only=True):

    constraints = []
    for arg in args:
        constraints.append(arg)
    if long_only:
        constraints.append(weights >= 0)

    if leverage == 1:
        constraints.append(cp.sum(weights) == 1)

    elif leverage > 1:
        constraints.append(cp.norm(weights, 1) <= leverage)
    else:
        raise ValueError('Leverage should be >= 1')
    return constraints


# To be used later
class ConstraintsBuilder:

    def __init__(self):
        self._constraints = []

    def long_only(self, weights):
        self._constraints.append(weights >= 0)

    def leverage(self, weights, level=1):
        if leverage == 1:
            constraints.append(cp.sum(weights) == 1)

        elif leverage > 1:
            constraints.append(cp.norm(weights, 1) <= leverage)
        else:
            raise ValueError('Leverage should be >= 1')

    def upper_bound_constraint(self, var, max_var):
        self._constraints.append(var <= max_var)

    def lower_bound_constraint(self, var, min_var):
        self._constraints.append(min_var <= var)

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, *args, **kwargs):
        raise ValueError('Cannot set constraints values')
