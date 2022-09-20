#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Rate Bond module
"""

from fdates.daycounter import Actual365A
from fdates.conventions import modified_following

from fixedincome.instruments.bond import Bond


class FixedRateBond(Bond):

    name = 'Fixed Rate Bond'

    def __init__(
            self, settlement, maturity, coupon=0, face_value=100,
            coupon_frequency=2,
            day_counter=Actual365A,
            business_day_convention=modified_following,
            end_of_month=0,
            ex_dividend=7,
            issue=None,
            first_coupon=None,
            last_coupon=None,
            holidays=None,
            format_='%Y-%m-%d'):
        super().__init__(
            settlement=settlement,
            maturity=maturity,
            coupon=coupon,
            face_value=face_value,
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
