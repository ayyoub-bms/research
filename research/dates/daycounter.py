"""
Module that contains the implementation of the different day count conventions.
"""
# pylint: disable=too-many-arguments


from abc import ABCMeta, abstractmethod

from .utils import as_date


class DayCounter:
    """Abstract Base class for managing Day count conventions

    Parameters
    ==========

    start: date, str, list-like of dates
        The start date(s). Note that if start is a list it must be of the same
        lenght as the end_date unless the end date is a date or a string repr
        of a date

    end:  date, str, list-like of dates
        The end date(s). Note that if end is a list it must be of the same
        lenght as the start date unless the start date is a date or a string
        repr of a date

    eom: Boolean
        Whether to apply the end of month rule or not
    """
    __metaclass__ = ABCMeta

    def __init__(self, start=None, end=None, eom=False):
        self.start = as_date(start)
        self.end = as_date(end)
        self.eom = eom

    @abstractmethod
    def days(self):
        """Returns the number of days between start and end dates"""
        raise NotImplementedError

    @abstractmethod
    def year(self):
        """Returns the number of days a year"""
        raise NotImplementedError

    def fraction(self):
        """Returns the fraction of years between start and end dates"""
        return self.days().days / self.year()


class Thirty360Base(DayCounter):
    """An abstract base class for day count conventions with 360 days a year"""
    __metaclass__ = ABCMeta

    def year(self):
        return 360

    @abstractmethod
    def days(self):
        """Returns the number of days between start and end dates"""
        raise NotImplementedError


class Actual360(Thirty360Base):
    """Implementation of the Actual 360 day count convention
    This is definition 4.16(e) in 2006 ISDA Definitions.
    This is the most used day count convention for money market instruments
    (maturity below one year).
    This day count is also called Money Market basis, Actual 360, or French
    """

    name = 'ACT/360'

    def days(self):
        return self.end - self.start


class Actual365F(DayCounter):
    """Implementation of the Actual 365F day count convention
    This is definition 4.16(d) in 2006 ISDA Definitions.
    The number 365 is used even in a leap year.
    This convention is also called English Money Market basis.
    """

    name = 'ACT/365 FIXED'

    def days(self):
        return self.end - self.start

    def year(self):
        return 365


# TODO: Use to implement the other 360 based day counters
def _days_impl(y_2, y_1, m_2, m_1, d_2, d_1):
    return 360 * (y_2 - y_1) + 30 * (m_2 - m_1) + (d_2 - d_1)
