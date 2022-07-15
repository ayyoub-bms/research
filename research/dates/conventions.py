"""
Module containing the different business day conventions
"""

from .utils import is_eom, roll


def modified_following(date, holidays=None):
    """Returns the next business day unless the latter occurs at the begining
    of next month, then the previous business day is returned

    Parameters
    ----------

    date : datetime object
        The date of interest

    holidays : list like of datetime objects
        The holidays of the previous / next days from the argument `date`

    Returns
    -------
        a datetime object compliant with the modified following rule
    """
    period = _period_holidays(date, holidays)
    adjusted_date = date
    # Move forward in time
    while adjusted_date.weekday() in (5, 6) or date in period:
        adjusted_date = roll(adjusted_date, days=1)
        if adjusted_date.month > date.month:
            break
    # Move backward in time
    return _adjust(date,  period=period, inc=False)


def modified_bimonthly(date, holidays=None):
    """Returns the next business day unless the latter is the end of month or
    in the 15th of the month, then the previous business day is returned

    Parameters
    ----------

    date : datetime object
        The date of interest

    holidays : list like of datetime objects
        The holidays of the previous / next days from the argument `date`

    Returns
    -------
        a datetime object compliant with the modified bimonthly rule
    """
    adjusted_date, period = _adjust(date, holidays)
    if adjusted_date.day == 15 or is_eom(adjusted_date):
        adjusted_date, _ = _adjust(date, period=period, inc=False)
    return adjusted_date


def following(date, holidays=None):
    """ Returns the next business day

    Parameters
    ----------

    date : datetime object
        The date of interest

    holidays : list like of datetime objects
        The holidays of the previous / next days from the argument `date`

    Returns
    -------
        a datetime object
    """
    return _adjust(date, holidays)


def preceding(date, holidays=None):
    """Returns the previous business day

    Parameters
    ----------

    date : datetime object
        The date of interest

    holidays : list like of datetime objects
        The holidays of the previous / next days from the argument `date`

    Returns
    -------
        a datetime object
    """
    return _adjust(date, holidays, inc=False)


def _adjust(date, holidays=None, period=None, inc=True):
    if period is None:
        period = _period_holidays(date, holidays)
    adjusted_date = date
    while adjusted_date.weekday() in (5, 6) or date in period:
        adjusted_date = roll(adjusted_date, days=(2*inc -1))
    return adjusted_date, period


def _period_holidays(date, holidays):
    period = []
    if holidays is not None:
        rel_start = roll(date, days=-10)
        rel_end = roll(date, days=10)
        period = [d for d in holidays if rel_start <= d <= rel_end]
    return period
