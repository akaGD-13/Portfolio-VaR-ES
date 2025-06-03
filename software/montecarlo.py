import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_var(prices: pd.Series, var_level: float,
                window_days: int, n_sims: int) -> pd.Series:
    """
    5-day VaR at var_level via Monte Carlo GBM simulation,
    parameters estimated over window_days.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    var = pd.Series(index=prices.index, dtype=float)
    for i in range(window_days, len(log_ret)):
        date = log_ret.index[i]
        data = log_ret.iloc[i - window_days : i]
        mu = data.mean()
        sigma = data.std()
        mean5 = 5 * mu
        std5 = np.sqrt(5) * sigma
        sims = np.random.normal(mean5, std5, size=n_sims)
        S = prices.loc[date]
        losses = -S * (np.exp(sims) - 1)
        var.loc[date] = np.percentile(losses, 100 * var_level)
    return var.dropna()


def compute_es(prices: pd.Series,
               es_level: float,
               window_days: int,
               n_sims: int) -> pd.Series:
    """
    5-day ES at es_level via Monte Carlo GBM simulation,
    parameters estimated over window_days.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    es = pd.Series(index=prices.index, dtype=float)

    # For each date, estimate mu/sigma on the past window_days,
    # simulate n_sims 5-day log-returns, convert to dollar losses,
    # then average the worst es_level tail.
    for i in range(window_days, len(log_ret)):
        date = log_ret.index[i]
        data = log_ret.iloc[i - window_days : i]
        mu, sigma = data.mean(), data.std()

        # simulate 5-day log-returns
        sims = np.random.normal(5*mu, np.sqrt(5)*sigma, size=n_sims)

        # convert to dollar losses
        S = prices.loc[date]
        losses = -S * (np.exp(sims) - 1)

        # find the cutoff at the es_level percentile (e.g. 99th)
        cutoff = np.percentile(losses, 100 * es_level)

        # average *only* the losses in the worst (1 − es_level) tail
        tail_losses = losses[losses >= cutoff]
        es.loc[date] = tail_losses.mean() if len(tail_losses) else np.nan

    return es.dropna()
