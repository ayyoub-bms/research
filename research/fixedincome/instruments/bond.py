"""
Bond instrument module
"""

from abc import ABCMeta

from research.dates.daycounter import Actual365A

from research.fixedincome.cashflow.cashflows import CashFlows
from research.fixedincome.cashflow.schedule import Schedule


class Bond:
    __metaclass__ = ABCMeta

    def __init__(
            self,
            settlement,
            maturity,
            coupon=0,
            face_value=100,
            coupon_frequency=2,
            day_counter=Actual365A,
            business_day_convention=None,
            end_of_month=0,
            ex_dividend=7,
            currency='EUR',
            issue=None,
            first_coupon=None,
            last_coupon=None,
            holidays=None,
            format_='%Y-%m-%d'):

        self.currency = currency
        self.coupon_rate = coupon
        self.face_value = face_value
        self.cash_flows = CashFlows(self)
        self.schedule = Schedule(
            settlement=settlement,
            maturity=maturity,
            issue=issue,
            first_coupon=first_coupon,
            last_coupon=last_coupon,
            coupon_frequency=coupon_frequency,
            holidays=holidays,
            day_counter=day_counter,
            business_day_convention=business_day_convention,
            end_of_month=end_of_month,
            ex_dividend=ex_dividend,
            format_=format_
        )

    def has_ofc(self):
        """Returns whether the bond as a first odd coupon date"""
        return self.schedule.first_coupon is not None

    def has_olc(self):
        """Returns whether the bond as a last odd coupon date"""
        return self.schedule.last_coupon is not None

    def is_zc(self):
        """Returns whether the bond is a zero coupon bond"""
        return self.coupon_rate == 0

    def regular_coupon_amount(self):
        return (
            self.face_value *
            self.coupon_rate /
            self.schedule.coupon_frequency
        )
