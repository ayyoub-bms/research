"""
Yield curve api module implements the following Yield Curves:
    SpotYieldCurve
    ForwardYieldCurve
    SwapYieldCurve
    ParYieldCurve
"""

# pylint: disable=import-error

import numpy as np
import pandas as pd

from research.dates.daycounter import Actual360
from research.dates.utils import today

from ..utils import futures_resets, futures_reset, swap_cashflow_dates
from ._generic import YieldCurve, Curve, _interpolate


def interpolate(df_rates, spot_date=today, daycounter=Actual360):
    """Interpolate the missing values of a yield curve"""
    _interpolate(df_rates, spot_date, daycounter)


class DiscountCurve(Curve):
    """Creates an instance of a discount curve

    Parameters
    ==========

        discounts: Pandas DataFrame with maturities as index.
            Contains discount factors

        spot_date: datetime
            the evaluation date

        daycounter: fdates.daycounter class
            The day count convention to be used : default (Act/360)
    """
    def __init__(self,
                 discounts,
                 spot_date=today,
                 daycounter=Actual360):

        super().__init__('Discount', discounts, spot_date, daycounter)
        self.__spots = self.__to_spots()

    def fit_terms(self, maturities):
        self.__fit_spots(maturities)
        self.__fit_discounts()

    @classmethod
    def from_spots(cls, df_spots, spot_date=today, daycounter=Actual360):
        """Bootstrap discount factors from spot rates

        Parameters
        ==========
            df_spots: pandas DataFrame with maturities in index as datetime
                the DataFrame containing the futures rates

            spot_date: datetime
                the evaluation date

            daycounter: fdates.daycounter class
                The day count convention to be used : default (Act/360)
        """
        terms = daycounter(spot_date, df_spots.index).fraction()
        rates = 1 / (1 + df_spots.mul(terms, axis=0))
        return cls(rates, spot_date, daycounter)

    @classmethod
    def from_pseudo_inverse(cls,
                            df_cashflows,
                            prices,
                            spot_date=today,
                            daycounter=Actual360):
        """Construct the discount curve using the pseudo inverse method

        Parameters
        ==========
            df_cashflows: pandas DataFrame with cashflow dates in columns with
                          increasing order and instruments id in index
                The DataFrame containing the cashflows for each instrument

            prices: list like
                the price of each security.

            spot_date: datetime
                the evaluation date

            daycounter: fdates.daycounter class
                The day count convention to be used : default (Act/360)
        """
        cash = df_cashflows.values
        date = df_cashflows.columns
        prices = np.asarray(prices)
        terms = daycounter(spot_date, date).fraction().to_series()
        discounts = _invert(cash, terms, prices)
        return cls(
            pd.DataFrame(discounts, index=date, columns=['discount']),
            spot_date,
            daycounter
        )

    def bootstrap_spots(self, df_spots):
        """Bootstrap discount factors from spot rates

        Parameters
        ==========
            df_spots: pandas DataFrame with maturities in index as datetime
                the DataFrame containing the futures rates
        """
        long, short = self.__discount(df_spots)

        if not short.empty:
            self._data = self._data.combine_first(short)

        if not long.empty:
            self._data = self._data.combine_first(long)

        self.__spots = df_spots.combine_first(self.__spots)

    def bootstrap_futures(self, df_futures):
        """Bootstrap discount factors from futures rates

        Parameters
        ==========
            df_futures: pandas DataFrame with maturities in index as datetime
                the DataFrame containing the futures rates
        """
        bootstrap_dates = df_futures.index
        cursor = self._data.index[-1]  # Last available discount date
        reset_dates = futures_resets(bootstrap_dates, cursor)
        for date in bootstrap_dates:
            rate = df_futures.loc[date]
            self.__bootstrap_futures(rate, date)
            ranges = (reset_dates < date) & (reset_dates > cursor)
            self.__fit_spots(reset_dates[ranges])
            cursor = date

        self.__fit_discounts()

    def bootstrap_swaps(self, df_swaps):
        """Bootstrap discount factors from swap rates

        Parameters
        ==========
            df_swaps: pandas DataFrame with maturities in index
                the DataFrame containing the futures rates
            periods: integer or list-like of integers
                The frequency of the swap cashflows:
                    1 for yearly
                    2 for bi-monthly
                    .. and so on
        """
        bootstrap_dates = df_swaps.index
        for date in bootstrap_dates:
            cf_dates = swap_cashflow_dates([date], self.spot_date)
            cf_dates.discard(self.spot_date)
            cf_dates = sorted(cf_dates)
            first_date = cf_dates[0]
            df_swaps = df_swaps.combine_first(pd.DataFrame(index=cf_dates))
            self.fit_terms([first_date])
            discount = self.get(first_date)
            term = self.daycounter(self.spot_date, first_date).fraction()
            df_swaps.loc[first_date] = (1/discount - 1) / term
            interpolate(df_swaps, self.spot_date, self.daycounter)
            self.__bootstrap_swaps(df_swaps, cf_dates)

    def to_spot(self):
        """Convert discount factors to zero coupon rates"""
        return SpotYieldCurve(df_rates=self.__spots,
                              spot_date=self.spot_date,
                              daycounter=self.daycounter,
                              discounts=self)

    def to_forward(self):
        """Convert discount factors to forward rates"""
        terms = self._compute_terms()
        first_term = terms.iloc[0]
        first_value = (1 / self._data.iloc[0] - 1) / first_term

        diff = terms.diff().fillna(first_term)
        df_forwards = self._data.shift(1).div(self._data, axis=0)

        df_forwards = df_forwards.sub(1).div(diff, axis=0)
        df_forwards.iloc[0] = first_value

        return ForwardYieldCurve(df_forwards,
                                 self,
                                 self.spot_date,
                                 self.daycounter)

    def to_swap(self):
        """Convert discount factors to swap rates"""
        df_swaps = self._data.copy()
        for date in df_swaps.index:
            term = self.daycounter(self.spot_date, date).fraction()
            if term <= 1:
                df_swaps.loc[date] = self.__spots.loc[date]
            else:
                cf_dates = swap_cashflow_dates([date], self.spot_date)
                cf_dates.discard(self.spot_date)
                cf_dates = sorted(cf_dates)
                self.fit_terms(cf_dates)
                rate = 1 - self._data.loc[date]
                terms = self._compute_terms(cf_dates)
                terms = terms.diff().fillna(terms.iloc[0])
                cumul = self._data.loc[cf_dates].mul(terms, axis=0).sum()
                df_swaps.loc[date] = rate / cumul

        return SwapYieldCurve(
            df_swaps,
            self,
            self.spot_date,
            self.daycounter
        )

    def __fit_discounts(self):
        long, short = self.__discount(self.__spots)
        self._data = self._data.combine_first(short).combine_first(long)

    def __fit_spots(self, maturities):
        df_maturities = pd.DataFrame(index=maturities)
        self.__spots = self.__spots.combine_first(df_maturities)
        interpolate(self.__spots, self.spot_date, self.daycounter)

    def __discount(self, df_spots):
        terms = self._compute_terms(df_spots.index)
        short_maturities = terms <= 1

        rates = df_spots.loc[short_maturities]
        short_discount = 1 / (1 + rates.mul(terms[short_maturities], axis=0))

        rates = df_spots.loc[~short_maturities]
        long_discount = 1 / ((1 + rates).pow(terms[~short_maturities], axis=0))

        return short_discount, long_discount

    def __to_spots(self, df_spots=None):
        if df_spots is None:
            df_spots = self._data.copy()

        terms = self._compute_terms(df_spots.index)
        short_maturities = terms <= 1

        # Compute short term zero rates
        rates = df_spots.loc[short_maturities]
        short_terms = terms[short_maturities]
        df_spots.loc[short_maturities] = (1/rates - 1).div(short_terms, axis=0)

        # Compute medium / long term zero rates
        rates = df_spots.loc[~short_maturities]
        fraction = 1 / terms[~short_maturities]
        df_spots.loc[~short_maturities] = (1 / rates).pow(fraction, axis=0) - 1
        return df_spots

    def __compute_spot_rate(self, date, discount_rate):
        term = self.daycounter(self.spot_date, date).fraction()
        if term <= 1:
            rate = (1 / discount_rate - 1) / term
        else:
            rate = (1 / discount_rate) ** (1 / term) - 1
        return rate

    def __bootstrap_futures(self, rate, date):
        reset = futures_reset(date)
        self.fit_terms([reset])
        term = self.daycounter(reset, date).fraction()
        discount = self.get(reset) / (1 + term * rate)
        self._data.loc[date] = discount
        self.__spots.loc[date] = self.__compute_spot_rate(date, discount)

    def __bootstrap_swaps(self, df_swaps, cf_dates):
        cumul = 0
        previous_date = self.spot_date
        discounts = [0]*len(cf_dates)
        i = 0
        for cf_date in cf_dates:
            diff = self.daycounter(previous_date, cf_date).fraction()
            rate = df_swaps.loc[cf_date]
            discount = (1 - cumul * rate) / (1 + rate * diff)
            discounts[i] = discount
            cumul = cumul + discount * diff
            previous_date = cf_date
            i = i + 1
        discounts = pd.DataFrame(discounts, index=cf_dates)
        self._data = self._data.combine_first(discounts)
        self.__spots = self.__to_spots(discounts).combine_first(self.__spots)


class SpotYieldCurve(YieldCurve):
    """Creates an instance of a zero coupon yield curve
    """

    def __init__(self,
                 df_rates,
                 spot_date=today,
                 daycounter=Actual360,
                 discounts=None):

        super().__init__('Spot', df_rates, spot_date, daycounter)

        if discounts is None:
            self._discounts = DiscountCurve.from_spots(self._data,
                                                       spot_date,
                                                       daycounter)
        else:
            self._discounts = discounts
            self._discounts.bootstrap_spots(df_rates)

    def fit_terms(self, maturities):
        frame = pd.DataFrame(index=maturities)
        self._data = self._data.combine_first(frame)
        _interpolate(self._data, self.spot_date, self.daycounter)
        self.discounts.bootstrap_spots(self._data.loc[maturities])

    def _compute_cashflows(self, series, dates):
        terms = self._compute_terms(dates)
        short, longs = terms <= 1, terms > 1
        matrix = np.diag(
            series.loc[dates][short].dropna()
            .mul(terms[short], axis=0)
            .add(1)
            .append(
                series.loc[dates][longs].dropna()
                .add(1)
                .pow(terms[longs], axis=0)
            )
        )

        df_cashflows = pd.DataFrame(matrix, columns=dates)
        df_cashflows['prices'] = 1
        return df_cashflows


class ForwardYieldCurve(YieldCurve):
    """Forward yield curve class definition"""

    def __init__(self,
                 df_rates,
                 df_discounts,
                 spot_date=today,
                 daycounter=Actual360):
        """Creates an instance of a forward yield curve

        Parameters
        ==========
            df_discounts: Pandas dataframe with maturities as index
                the discount factors for all the reset dates of df_rates
        """
        super().__init__('Forward', df_rates, spot_date, daycounter)
        self._discounts = df_discounts

    def _compute_cashflows(self, series, dates):
        """Build Cashflow matrix from futures securities maturing
        in `futures_dates`
        """
        rolls = pd.DatetimeIndex(map(futures_reset, dates))

        df_cashflows = pd.DataFrame(
            columns=rolls.union(dates),
            index=list(range(len(dates))),
            dtype=np.float
        )

        i = 0
        for date, reset in zip(dates, rolls):
            df_cashflows.loc[i, reset] = -1
            diff = (date - reset).days / 360
            df_cashflows.loc[i, date] = 1 + series.loc[date] * diff
            i += 1

        df_cashflows['prices'] = 0
        return df_cashflows.fillna(0)


class SwapYieldCurve(YieldCurve):
    """Create an instance of a Swap yield curve
    """

    def __init__(self,
                 df_rates,
                 discount_curve,
                 spot_date=today,
                 daycounter=Actual360):

        super().__init__('Swap', df_rates, spot_date, daycounter)
        self._discounts = discount_curve

    def _compute_cashflows(self, series, dates):
        df_cashflows = pd.DataFrame(dtype=np.float, index=range(len(dates)))
        i = 0
        prices = [0] * len(dates)

        for date in dates:
            rate = series.loc[date]
            cf_dates = sorted(swap_cashflow_dates([date], self.spot_date))
            effective_dates = cf_dates[1:-1]
            terms = self._compute_terms(cf_dates)
            terms = terms.diff().fillna(terms.iloc[0])

            df_ = pd.DataFrame(
                [rate * terms[effective_dates]],
                index=[i],
                columns=effective_dates
            )

            df_cashflows = df_.combine_first(df_cashflows)

            if cf_dates[0] == self.spot_date:
                prices[i] = 1

            else:
                df_cashflows.loc[i, cf_dates[0]] = -1
            df_cashflows.loc[i, cf_dates[-1]] = 1 + rate * terms[cf_dates[-1]]
            i = i + 1

        df_cashflows['prices'] = prices
        return df_cashflows.fillna(0)


def _invert(cash, terms, prices):
    terms = np.sqrt(terms.diff().fillna(terms.iloc[0]).values)

    one = np.zeros(len(terms)).transpose()
    one[0] = 1

    w_inverse = np.diag(terms)
    m_inverse = np.tril([1] * len(terms))
    mat = cash.dot(m_inverse.dot(w_inverse))
    m_t = mat.transpose()
    mmt_inverse = np.linalg.inv(mat.dot(m_t))
    delta = m_t.dot(mmt_inverse).dot(prices - cash.dot(m_inverse.dot(one)))
    return m_inverse.dot(w_inverse.dot(delta) + one)
