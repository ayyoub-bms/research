import pandas as pd


def drift_weights(returns: pd.DataFrame,
                  weights: pd.DataFrame or pd.Series,
                  freq='1M'):
    """ The weights are supposed to contain as index the rebalancing dates and
    as columns the corresponding weights.  For example, if the allocations are
    computed the 31-Jan at night and effective the 01-Feb then the index of the
    weights should be set to the the 1st of Feb and the first date of the
    return series should be the 31-Jan. The index frequency of the weights
    should match the rebalancing frequency `freq`

    Example:
    --------
    Returns series
    ++++++++++++++++++++++++++++++++++++++++++
    |31-01-1999| xxx | xxx | ... | xxx | xxx |
    |01-02-1999| xxx | xxx | ... | xxx | xxx |
    |02-01-1999| xxx | xxx | ... | xxx | xxx |
    |..........|.....|.....|.....|.....|.....|
    |01-01-2020| xxx | xxx | ... | xxx | xxx |
    ++++++++++++++++++++++++++++++++++++++++++
    Weights series
    ++++++++++++++++++++++++++++++++++++++++++
    |01-02-1999| xxx | xxx | ... | xxx | xxx |
    |01-03-1999| xxx | xxx | ... | xxx | xxx |
    |..........|.....|.....|.....|.....|.....|
    |01-01-2020| xxx | xxx | ... | xxx | xxx |
    ++++++++++++++++++++++++++++++++++++++++++

    Returns:
    --------
        Drifted weights as a Pandas DataFrame with the same frequency as the
        returns.
    """
    drifted = pd.DataFrame(index=returns.index, columns=returns.columns)

    if isinstance(weights, pd.Series):
        drifted.update(
            returns
            .groupby(pd.Grouper(freq=freq))
            .apply(period_drift, weights=weights)
        )
    else:
        n = weights.shape[0]
        for i in range(n-1):
            curr_rebal_date, next_rebal_date = weights.index[i:i+2]
            block = returns.loc[curr_rebal_date:next_rebal_date]
            drifted.update(period_drift(block, weights.loc[curr_rebal_date]))
        curr_rebal_date = weights.index[-1]
        block = returns.loc[curr_rebal_date:]
        drifted.update(period_drift(block, weights.loc[curr_rebal_date]))
    return drifted


def normalize(w):
    if isinstance(w, pd.Series):
        return w / w.sum()
    return w.div(w.sum(axis=1), axis=0)


def period_drift(returns: pd.DataFrame, weights: pd.Series):
    drifted = pd.DataFrame(index=returns.index, columns=returns.columns)
    drifted.iloc[0] = weights
    N = returns.shape[0]
    for i in range(1, N):
        numer = drifted.iloc[i - 1, :] * (1 + returns.iloc[i, :])
        w = numer / numer.sum()
        drifted.iloc[i, :] = w
    return drifted
