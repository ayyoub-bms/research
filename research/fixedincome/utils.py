"""Helper module for fixed income utilities

"""

# pylint: disable=import-error

import numpy as np

from dates import utils as _ut


def futures_reset(dates):
    """Returns the reset date of futures with maturity date `dates`"""
    return _ut.imm_date(_ut.roll(_ut.bom(dates), months=-3))


def futures_resets(futures_dates, spot_date):
    """Finds all the reset dates of the corresponding to `futures_dates` down
    to the last available spot rate `spot_date`
    """
    dates = set()
    for date in futures_dates:
        while date > spot_date:
            date = futures_reset(date)
            dates.add(date)

    return np.asarray(sorted(dates))


def swap_cashflow_dates(swap_dates, spot_date, period=1):
    """Retrun all the swap cashflow dates for the `swap_dates` starting from
    the `spot_date` with a total of `period`  payments a year.
    """
    if isinstance(period, int):
        period = [period]*len(swap_dates)

    dates = set()
    i = 0
    for date in swap_dates:
        while date >= spot_date:
            dates.add(date)
            date = _ut.roll(date, months=-int(12/period[i]))
        i = i + 1
    return dates


def adjust_coupon_days(coupon_dates, ref):
    """Adjust coupon days"""
    mod = False
    if not isinstance(coupon_dates, list):
        coupon_dates = [coupon_dates]
        mod = True

    if _ut.is_eom(ref):
        coupon_dates = [_ut.eom(t) for t in coupon_dates]
    elif ref.day <= 28:
        coupon_dates = [t.replace(day=ref.day) for t in coupon_dates]
    else:
        coupon_dates = [
            t.replace(
                day=ref.day if t.month != 2 else _ut.eom(t).day
            ) for t in coupon_dates
        ]
    if mod:
        return coupon_dates[0]
    return coupon_dates


def discount_from_rate(term_to_maturity,
                       rate,
                       compounding='compounded',
                       period=1):
    """Returns the discount factor of a given period expressed in years:
        `term_to_maturity`

    Parameters
    ==========
        term_to_maturity : float
            The period between the evaluation date and the maturity expressed
            in years.

    """
    interest = rate_from_discount(term_to_maturity, rate, compounding, period)
    return 1 / (1 + interest)


def rate_from_discount(term_to_maturity,
                       rate,
                       compounding='compounded',
                       period=1):
    """Returns the discount factor of a given period expressed in years:
        `term_to_maturity`

    Parameters
    ==========
        term_to_maturity : float
            The period between the evaluation date and the maturity expressed
            in years.

    """
    assert(compounding in ['simple', 'compounded', 'continuous'])

    if compounding == 'compounded':
        interest = _interest_from_compounded_rate(term_to_maturity,
                                                  rate,
                                                  period)

    elif compounding == 'simple':
        interest = _interest_from_simple_rate(term_to_maturity, rate)

    else:
        interest = _interest_from_continuous_rate(term_to_maturity, rate)
    return interest


def _interest_from_simple_rate(term_to_maturity, rate):
    """Computes the interest rate to be earned on the period

    1 + R(t, T) = 1 + (T - t) * L(t, T)

    Parameters
    ----------
    term_to_maturity : float
        The fraction of the year represeting the short interest period
    rate : float
        The rate applicable to the period (Libor for example)
    Returns
    -------
        R(t, T)
    """
    return term_to_maturity * rate


def _interest_from_compounded_rate(term_to_maturity, rate, period=1):
    """Computes the interest rate to be earned on the period

    1 + R(t, T) = (1 + y_n(t, T)/n)^(n(T-t))

    Parameters
    ----------
    term_to_maturity : float
        The fraction of the year represeting the short interest period
    rate : float
        The yield / the continuously compounded rate.
    Returns
    -------
        R(t, T)
    """
    return (1 + rate / period) ** (period * term_to_maturity) - 1


def _interest_from_continuous_rate(term_to_maturity, rate):
    """Computes the interest rate to be earned on the period

    1 + R(t, T) = exp((T - t) * y(t, T))

    Parameters
    ----------
    term_to_maturity : float
        The fraction of the year represeting the short interest period
    rate : float
        The yield / the continuously compounded rate.
    Returns
    -------
        R(t, T)
    """
    return np.exp(term_to_maturity * rate) - 1
