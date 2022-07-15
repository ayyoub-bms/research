"""
Yield Curve implementation for various term structures
"""


from abc import ABCMeta, abstractmethod


class Curve:
    """Generic cuve abstract calss"""
    __metaclass__ = ABCMeta

    def __init__(self, curve_type, df_rates, spot_date, daycounter):

        self._data = df_rates
        self.spot_date = spot_date
        self.curve_type = curve_type
        self.daycounter = daycounter

    @property
    def term_structure(self):
        """Returns the term strucutre dataframe"""
        return self._data

    def get(self, maturities):
        """Returns the term structure by interpolating the yield on the given
        `dates`

        Parameters
        ==========

            maturities: List like of datetime
                The new maturities to be added for interpolation
        """
        return self._data.loc[maturities]

    @abstractmethod
    def fit_terms(self, maturities):
        """Returns the term structure by interpolating the yield on the given
        `maturities`

        Parameters
        ==========

            maturities: List like of datetime
                The new maturities to be added for interpolation
        """
        raise NotImplementedError('Please implement in derived classes')

    def plot(self):
        """Plot the term structure"""
        terms = self._compute_terms()
        axes = self._data.set_index(terms).mul(100).plot()
        axes.set_xlabel('term to maturity (in years)')
        axes.set_ylabel('{} yield (in %)'.format(self.curve_type))
        axes.set_title('{} Yield Curve'.format(self.curve_type))
        return axes

    def _compute_terms(self, dates=None):
        if dates is None:
            dates = self._data.index
        terms = self.daycounter(self.spot_date, dates).fraction().to_series()
        terms.index = dates
        return terms

    def __repr__(self):
        return '\n'.join((
            '    Curve type: ' + self.curve_type,
            'Term structure:',
            '',
            str(self._data)
        ))


class YieldCurve(Curve):
    """Creates an instance of a yield curve

    Parameters
    ==========

        curve_type: str
            The curve type one of ('spot', 'par', 'forward', 'swap')

        spot_date: datetime
            the evaluation date

        df_rates: Pandas DataFrame with maturities as index.
            Contains either zero coupon rates, par, forward or swap rate.

        daycounter: fdates.daycounter class
            The day count convention to be used : default (Act/360)

    """
    __metaclass__ = ABCMeta

    def __init__(self, curve_type, df_rates, spot_date, daycounter):
        super().__init__(curve_type, df_rates, spot_date, daycounter)
        self._discounts = None

    @property
    def discounts(self):
        """Returns the discount curve instance"""
        return self._discounts

    def fit_terms(self, maturities):
        spot_curve = self.to_spot()
        spot_curve.build_points(maturities)
        method_name = '_'.join(('to', self.curve_type.lower()))
        curve = getattr(spot_curve.discounts, method_name)()
        self._data = curve.term_structure

    def cashflows(self, dates):
        """Build Cashflow matrix for each yield curve"""
        return {
            col: self._compute_cashflows(self._data[col], dates)
            for col in self._data.columns
        }

    def discount_factors(self):
        """Computes the discount factors
        """
        return self._discounts.term_structure

    def to_swap(self):
        """Convert the current yield curve to a swap curve"""
        return self._discounts.to_swap()

    def to_spot(self):
        """Convert the current yield curve to a zero coupon curve"""
        return self._discounts.to_spot()

    def to_forward(self):
        """Convert the current yield curve to a forward yield curve"""
        return self._discounts.to_forward()

    @abstractmethod
    def _compute_cashflows(self, series, dates):
        """Build Cashflow matrix"""
        raise NotImplementedError('Please implement in derived classes')


def _interpolate(df_rates, spot_date, daycounter):
    """Interpolate the missing values of a yield curve"""
    index = df_rates.index
    terms = daycounter(spot_date, df_rates.index).fraction()
    df_rates.index = terms
    df_rates.interpolate(method='index',
                         limit_direction='both',
                         inplace=True)
    df_rates.index = index
