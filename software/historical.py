import pandas as pd
import numpy as np

def compute_var(prices: pd.Series,
                var_level: float,
                window_days: int) -> pd.Series:
    """
    5-day empirical VaR at var_level using 5-day log-returns
    and a rolling window of window_days, returned in dollars.
    """
    alpha = 1 - var_level
    # 1) 5-day log returns
    r5 = np.log(prices / prices.shift(5)).dropna()

    # 2) rolling quantile of r5 at alpha
    r_q = r5.rolling(window_days).quantile(alpha).dropna()

    # 3) convert to dollar loss
    dollar_var = prices.loc[r_q.index] * (1 - np.exp(r_q))
    return dollar_var

def compute_es(prices: pd.Series,
               es_level: float,
               window_days: int) -> pd.Series:
    """
    5-day empirical ES at es_level using 5-day log-returns
    and a rolling window of window_days, returned in dollars.
    """
    alpha = 1 - es_level
    r5 = np.log(prices / prices.shift(5)).dropna()

    def es_log(x):
        cutoff = np.percentile(x, 100 * alpha)
        tail   = x[x <= cutoff]
        return tail.mean() if len(tail) else np.nan

    # 1) rolling average of tail log-returns
    r_es = r5.rolling(window_days).apply(es_log, raw=False).dropna()

    # 2) convert to dollar ES
    dollar_es = prices.loc[r_es.index] * (1 - np.exp(r_es))
    return dollar_es
