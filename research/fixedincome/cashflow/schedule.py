"""
Bond schedule management module.
"""

import logging

from dateutil.relativedelta import relativedelta as _rl

import numpy as np
import pandas as pd

from research.fixedincome.utils import adjust_coupon_days
from research.dates.daycounter import Actual365A
from research.dates.conventions import modified_following

LOGGER = logging.getLogger(__name__)


class Schedule:
    def __init__(
            self,
            settlement,
            maturity,
            issue=None,
            first_coupon=None,
            last_coupon=None,
            holidays=None,
            coupon_frequency=2,
            day_counter=Actual365A,
            business_day_convention=modified_following,
            end_of_month=1,
            ex_dividend=7,
            format_=None):

        self.holidays = holidays
        self.ex_dividend = ex_dividend
        self.day_counter = day_counter
        self.end_of_mont_rule = end_of_month
        self.coupon_frequency = coupon_frequency
        self._issue = pd.to_datetime(issue, format=format_)
        self._maturity = pd.to_datetime(maturity, format=format_)
        self._settlement = pd.to_datetime(settlement, format=format_)
        self._last_coupon = pd.to_datetime(last_coupon, format=format_)
        self._first_coupon = pd.to_datetime(first_coupon, format=format_)

        self.business_day_convention = business_day_convention

        assert (self._settlement <= self._maturity)
        if issue is not None:
            assert (self._issue <= self._settlement)
        if last_coupon is not None:
            assert (self._last_coupon < self._maturity)
        if first_coupon is not None and self._issue is not None:
            assert (self._first_coupon > self._issue)

        self.time_step = int(12 / coupon_frequency)

    @property
    def settlement(self):
        return self._settlement

    @property
    def value_date(self):
        return self.business_day_convention(self._settlement, self.holidays)

    @settlement.setter
    def settlement(self, date):
        self._settlement = pd.to_datetime(date)

    @property
    def maturity(self):
        return self._maturity

    @maturity.setter
    def maturity(self, date):
        self._maturity = pd.to_datetime(date)

    @property
    def issue(self):
        return self._issue

    @issue.setter
    def issue(self, date):
        self._issue = pd.to_datetime(date)

    @property
    def first_coupon(self):
        return self._first_coupon

    @first_coupon.setter
    def first_coupon(self, date):
        self._first_coupon = pd.to_datetime(date)

    @property
    def last_coupon(self):
        return self._last_coupon

    @last_coupon.setter
    def last_coupon(self, date):
        self._last_coupon = pd.to_datetime(date)

    def quasi_coupon_dates(self):
        dates = self._all_quasi_dates()
        if self._settlement > dates[-1]:
            return []
        idx = next(i for i, x in enumerate(dates) if x >= self._settlement)
        return dates[idx:]

    def coupon_dates(self):
        dates = self.quasi_coupon_dates()

        fc = self._first_coupon
        lc = self._last_coupon
        if dates:
            if fc is not None and self.settlement < fc:
                while dates[0] != fc:
                    del dates[0]
                    if not dates:
                        break

            if lc is not None:
                while len(dates) > 1 and dates[-2] > lc:
                    del dates[-2]

        return sorted(dates)

    def quasi_issue_date(self):
        return self._all_quasi_dates()[0]

    def previous_quasi_coupon(self):
        dates = self._all_quasi_dates()
        idx = next(i for i, x in enumerate(dates) if x > self._settlement)
        return dates[idx - 1]

    def next_quasi_coupon(self):
        dates = self._all_quasi_dates()
        idx = next(i for i, x in enumerate(dates) if x > self._settlement)
        return dates[idx]

    def last_quasi_coupon(self):
        dates = self._all_quasi_dates()
        if len(dates) >= 2:  # there is still quasi coupons
            return dates[-2]
        else:  # return the last one = maturity
            return dates[-1]

    def first_coupon_date(self):
        if self._first_coupon is not None:
            return self._first_coupon
        else:
            return self.first_quasi_coupon_date()

    def first_quasi_coupon_date(self):
        qid = self.quasi_issue_date()
        return adjust_coupon_days(qid + _rl(months=self.time_step), qid)

    def previous_coupon(self):
        # Add a check for issues before settlement
        ncp = self.next_coupon()
        return adjust_coupon_days(ncp + _rl(months=self.time_step), ncp)

    def next_coupon(self):
        dates = self.coupon_dates()
        if dates:
            return dates[0]
        return None

    def ex_dividend_date(self):
        return self.business_day_convention(
            self.next_coupon() - _rl(days=self.ex_dividend),
            self.holidays
        )

    def full_odd_first_count(self):
        if self._first_coupon is not None:
            if self._issue is not None:
                count = 0
                tmp_date = self._first_coupon
                if tmp_date - _rl(months=self.time_step) < self._issue:
                    return 0
                while tmp_date > self._issue:
                    count += 1
                    tmp_date -= _rl(months=self.time_step)
                return count - 1
            else:
                return 1
        return 0

    def full_odd_last_count(self):
        if self._last_coupon is not None:
            dates = np.asarray(self._all_quasi_dates())
            return np.sum(dates > self._last_coupon) - 1
        return 0

    def _all_quasi_dates(self):
        dates = []
        has_issue = self._issue is not None
        has_fc = self._first_coupon is not None
        has_lc = self._last_coupon is not None

        if has_fc:
            if has_issue:
                # At this point we have either a long or short first coupon
                # we then go back up to a date before issue: the quasi issue
                # date
                first_date = self._first_coupon
                while first_date > self._issue:
                    first_date -= _rl(months=self.time_step)
            else:
                # The first date will be used to compute the accrued
                # and the first coupon
                first_date = self._first_coupon
                while first_date > self._settlement:
                    first_date -= _rl(months=self.time_step)

            # Now we generate the quasi dates based on the first coupon date
            last_date = first_date
            while last_date < self._maturity:
                dates.append(last_date)
                last_date += _rl(months=self.time_step)
            dates = adjust_coupon_days(dates, self._computation_date())
            # This is because we start from the first coupon
            # Thus we need to have the maturity in as last coupon date
            dates.append(self._maturity)
            return sorted(dates)

        # Now we handle the odd last coupon
        if has_lc:
            # Now we setup the quasi dates based on the last coupon date
            # Note that if we have a long last coupon
            last_date = self._last_coupon
            while last_date < self._maturity:  # handle long last coupons
                last_date += _rl(months=self.time_step)
            if last_date > self._maturity:  # exclude last coupon if > mat
                last_date -= _rl(months=self.time_step)

            if has_issue:
                first_date = self._issue
            else:
                first_date = self.settlement

            while last_date > first_date:
                dates.append(last_date)
                last_date -= _rl(months=self.time_step)
            dates.append(last_date)  # include the date before the first_date
            dates = adjust_coupon_days(dates, self._computation_date())
            # This is because we start from the last coupon
            # Thus we need to have the maturity in as last coupon date
            dates.append(self._maturity)
            return sorted(dates)

        # we set the regular bond coupon dates from maturity
        last_date = self._maturity
        if has_issue:
            first_date = self._issue
        else:
            first_date = self.settlement
        while last_date > first_date:
            dates.append(last_date)
            last_date -= _rl(months=self.time_step)
        dates.append(last_date)  # include the date before the first_date
        dates = adjust_coupon_days(dates, self._computation_date())
        return sorted(dates)

    def _computation_date(self):
        date = self._maturity
        if self._last_coupon is not None:
            date = self._last_coupon
        if self._first_coupon is not None:
            date = self._first_coupon
        return date

    def dates_fraction(self):
        dates = self.coupon_dates()
        fractions = np.arange(len(dates), dtype=np.float64)
        days = self.day_counter(self._settlement, dates[0]).days()
        period = self.day_counter(
            self.previous_quasi_coupon(), self.next_quasi_coupon()
        ).days()
        fractions[:-1] += days / period
        # The last fraction is computed based on the maturity date, thus we
        # need to roll back up to the maturity to compute the last fraction.
        date = self._maturity
        count = -1
        while date > self._settlement:
            count += 1
            date -= _rl(months=self.time_step)
        next_date = adjust_coupon_days(
            date + _rl(months=self.time_step),
            self._maturity
        )
        date = adjust_coupon_days(date, self._maturity)
        days = self.day_counter(self._settlement, next_date).days()
        period = self.day_counter(date, next_date).days()
        fractions[-1] = count + days / period
        # case where the settlement is not a coupon date
        if self._settlement not in dates:
            dates = [self._settlement] + dates

        # Formatting
        df_fractions = pd.DataFrame(index=dates, columns=['YEAR_FRACTIONS'])
        df_fractions.iloc[0, 0] = 0
        df_fractions.iloc[1:, 0] = fractions
        return df_fractions
