"""
Bond Cash flow manager module
"""

import logging


from dateutil.relativedelta import relativedelta as _rl

from ..utils import adjust_coupon_days

LOGGER = logging.getLogger(__name__)


class CashFlows:
    """A class representing a stream of cash flows of a bond.

    This class manages the computations related coupon and redemption values of
    a bond.
    """

    def __init__(self, bond):
        self.bond = bond

    def amounts(self):
        """Computes the cash flow amounts of the bond"""
        schedule = self.bond.schedule

        df_cash_flows = schedule.dates_fraction()
        df_cash_flows.name = 'CASH_FLOW_MATRIX'
        df_cash_flows.index.name = 'CF_DATES'

        odd_last = self.bond.has_olc()
        odd_first = self.bond.has_ofc()

        df_cash_flows['CF_AMOUNTS'] = self.bond.regular_coupon_amount()
        df_cash_flows = df_cash_flows[['CF_AMOUNTS', 'YEAR_FRACTIONS']]

        if not odd_last and not odd_first:
            self._compute_regular_amounts(df_cash_flows)
        elif odd_first:
            self._compute_first_amounts(df_cash_flows)
        else:
            # Case where there is an odd last coupon
            self._compute_last_amounts(df_cash_flows)

        df_cash_flows.iloc[-1, 0] += self.bond.face_value
        return df_cash_flows

    def _compute_first_amounts(self, df_cash_flows):

        schedule = self.bond.schedule
        issue = schedule.issue
        mat = schedule.maturity
        stl = schedule.settlement
        roll = _rl(months=schedule.time_step)
        sfc = schedule.first_coupon
        slc = schedule.last_coupon
        qid = schedule.quasi_issue_date()
        pcd = schedule.previous_quasi_coupon()
        ncd = schedule.next_quasi_coupon()
        cfa = self.bond.regular_coupon_amount()
        fqd = schedule.first_quasi_coupon_date()

        if stl < sfc and issue is not None:
            LOGGER.debug('computing for first odd coupon bond')
            # compute the first coupon amount if the settlement falls between
            # the issue date and the first coupon date and the issue date
            # is known. If the issue date is unknown that we consider the
            # period a full coupon period and thus nothing is to be done.

            # First Coupon amount
            lcv = cfa * schedule.full_odd_first_count()
            df_cash_flows.iloc[1, 0] = lcv + self._fraction(
                cfa, issue, fqd, qid, fqd
            )

            # Accrued
            # We get how many are they of full coupon dates between the issue
            # date and the settlement date
            count = 0
            coupon_date = stl
            while coupon_date >= issue:
                count += 1
                coupon_date -= roll
            count -= 1
            fixed_accrued = -count * cfa
            # We compute the accrued for the period between the settlement and
            # the previous quasi coupon
            variable_accrued = self._fraction(cfa, stl, pcd, pcd, ncd)
            # We compute the accrued for the period between the issue date and
            # the first quasi coupon
            variable_accrued += self._fraction(cfa, fqd, issue, qid, fqd)
            # We are good to go !
            df_cash_flows.iloc[0, 0] = variable_accrued + fixed_accrued

        elif stl < sfc and issue is None:
            LOGGER.debug('odd first before first coupon and no issue')
            df_cash_flows.iloc[0, 0] = self._fraction(cfa, stl, qid, qid, fqd)
        else:
            LOGGER.debug('odd first after first coupon')
            # If we don't have the issue date or if the settlement is
            # after the first coupon either way the computation of the
            # accrued is the same
            df_cash_flows.iloc[0, 0] = self._fraction(cfa, stl, pcd, pcd, ncd)

        lqc = schedule.last_quasi_coupon()
        df_cash_flows.iloc[-1, 0] = self._fraction(
            cfa, lqc, mat, lqc, adjust_coupon_days(lqc + roll, lqc)
        )

        if slc is not None:
            # We need to take care of the case where there is a last coupon
            flc = schedule.full_odd_last_count()
            df_cash_flows.iloc[-1, 0] = cfa * flc + self._fraction(
                cfa, lqc, mat, lqc, adjust_coupon_days(lqc + roll, lqc)
            )
            if stl >= slc:
                self._handle_after_last_coupon_accrued(df_cash_flows)

    def _compute_last_amounts(self, df_cash_flows):
        LOGGER.debug('computing for last odd coupon bond')

        schedule = self.bond.schedule
        issue = schedule.issue
        stl = schedule.settlement
        lc = schedule.last_coupon
        fcp = schedule.first_coupon_date()
        cfa = self.bond.regular_coupon_amount()
        pcp = schedule.previous_quasi_coupon()
        ncp = schedule.next_quasi_coupon()

        # Case where we are before the last coupon : if there is no issue
        # date then the fcp will always be the last coupon date thus we need
        #  to add an extra condition `or issue is None and stl < lc`
        if issue is not None and fcp < stl < lc or issue is None and stl < lc:
            LOGGER.debug('odd last with regular period')
            df_cash_flows.iloc[0, 0] = self._fraction(cfa, stl, pcp, pcp, ncp)

        # case where we are at a coupon date, no need for calculation, return 0
        elif pcp == stl:
            df_cash_flows.iloc[0, 0] = 0
        # case where we are before the first coupon date
        elif stl < fcp < lc:
            LOGGER.debug('odd last before first coupon')
            qid = schedule.quasi_issue_date()
            if issue is None:
                # Compute the accrued
                df_cash_flows.iloc[0, 0] = self._fraction(
                    cfa, stl, qid, qid, fcp)

            else:
                # Compute First Coupon
                cf = df_cash_flows.iloc[1, 0] = self._fraction(
                    cfa, issue, fcp, qid, fcp
                )
                # Compute the accrued
                df_cash_flows.iloc[0, 0] = self._fraction(
                    cf, stl, issue, issue, fcp
                )

        mat = schedule.maturity
        roll = _rl(months=schedule.time_step)
        flc = schedule.full_odd_last_count()
        lqc = adjust_coupon_days(lc + roll * flc, lc)
        # Compute Last Coupon
        df_cash_flows.iloc[-1, 0] = cfa * flc + self._fraction(
            cfa, lqc, mat, lqc, adjust_coupon_days(lqc + roll, lqc)
        )

        if stl > lc:
            flc = schedule.full_odd_last_count()
            lqc = adjust_coupon_days(lc + roll * flc, lc)
            df_cash_flows.iloc[-1, 0] = cfa * flc + self._fraction(
                cfa, lqc, mat, lqc, adjust_coupon_days(lqc + roll, lqc)
            )
            self._handle_after_last_coupon_accrued(df_cash_flows)

    def _compute_regular_amounts(self, df_cash_flows):

        schedule = self.bond.schedule
        issue = schedule.issue
        stl = schedule.settlement
        pcd = schedule.previous_quasi_coupon()
        ncd = schedule.next_quasi_coupon()
        cfa = self.bond.regular_coupon_amount()
        roll = _rl(months=schedule.time_step)
        # Case where the settlement is a coupon date
        if issue is not None and stl == pcd:
            LOGGER.debug('at coupon case')
            df_cash_flows.iloc[0, 0] = 0
        # If we are before the first coupon:
        elif issue is not None and stl - roll < issue:
            LOGGER.debug('before first coupon case')
            # cash flow amount
            cf = self._fraction(cfa, issue, ncd, pcd, ncd)

            # Accrued
            df_cash_flows.iloc[0, 0] = self._fraction(
                cf, stl, issue, issue, ncd)
            df_cash_flows.iloc[1, 0] = cf
        else:
            LOGGER.debug('anywhere else')
            df_cash_flows.iloc[0, 0] = self._fraction(cfa, stl, pcd, pcd, ncd)

    def _fraction(self, cash, days_start, days_end, period_start, period_end):
        schedule = self.bond.schedule
        days = schedule.day_counter(days_start, days_end).days()
        period_days = schedule.day_counter(period_start, period_end).days()
        return cash * days / period_days

    def _handle_after_last_coupon_accrued(self, df_cash_flows):
        LOGGER.debug('odd coupon after last coupon')
        schedule = self.bond.schedule
        cfa = self.bond.regular_coupon_amount()
        lc = schedule.last_coupon
        mat = schedule.maturity
        stl = schedule.settlement
        roll = _rl(months=schedule.time_step)

        # The tricky part is that we  need to compute the accrued based
        # on the maturity date
        count = -1
        date = mat
        while date >= stl:
            count += 1
            date -= roll

        date, rolled = adjust_coupon_days([date, date + roll], mat)
        variable_accrued = self._fraction(cfa, stl, date, date, rolled)
        count = -1
        while date >= lc:
            count += 1
            date -= roll

        fixed_accrued = -cfa * count
        date, rolled = adjust_coupon_days([date, date + roll], mat)
        variable_accrued += self._fraction(cfa, rolled, lc, date, rolled)
        df_cash_flows.iloc[0, 0] = variable_accrued + fixed_accrued

