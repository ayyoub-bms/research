"""
Dates utilities
"""

# pylint: disable=invalid-name

import datetime as dt

from dateutil.relativedelta import relativedelta as _rl
from pandas.tseries.offsets import to_datetime, WeekOfMonth

today = to_datetime(dt.date.today())


def imm_date(dates):
    """Returns IMM date: third Wednesday of March, June, September and December
    (i.e., between the 15th and 21st, whichever such day is a Wednesday) of
    the given date. IMM stands for the International Monetary Market.

    Parameters
    ==========
        date: str, datetime or list-like of str or datetimes
            The dates of interest
    """
    return to_datetime(dates) + WeekOfMonth(week=2, weekday=2)


def as_date(date):
    """Converts the `date` to pandas datetime format"""
    return to_datetime(date)


def roll(dates, days=0, months=0, years=0, **kwargs):
    """Roll the dates to a specified dates in the past or future"""
    dates = to_datetime(dates)
    rolling = _rl(days=days, months=months, years=years, **kwargs)
    try:
        return dates + rolling
    except TypeError:
        return dates.map(lambda date: date + rolling)


def eom(date):
    """Returns the end of month of a date"""
    next_month = date + dt.timedelta(days=28)
    return next_month - dt.timedelta(days=next_month.day)


def bom(date):
    """Returns the end of month of a date"""
    date = as_date(date)
    try:
        date = date - dt.timedelta(days=date.day-1)
    except TypeError:
        date = date.map(lambda x: x - dt.timedelta(days=x.day-1))
    return date


def is_eom(date):
    """Return whether the date is an end of month date"""
    return (date + dt.timedelta(days=1)).month == (date.month + 1)


def is_leap(date):
    """Check if the year of `date` is a leap year"""
    year = as_date(date).year
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def has_feb29(date_start, date_end):
    """Returns whether there is a 29th of February between `date_start` and
    `date_end`
    """
    if date_end.month == 2 and date_end.day == 29:
        return True

    if is_leap(date_end) and date_end.month > 2:
        return True

    nb_years = date_end.year - date_start.year
    if nb_years == 0:
        return False

    i = 1
    year = date_start.year
    while i < nb_years:
        if is_leap(year + i):
            return True
        i += 1

    return False
