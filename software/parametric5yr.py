# parametric5yr.py

import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_var(prices: pd.Series, var_level: float) -> pd.Series:
    """
    5-day VaR at var_level using GBM parameters estimated
    over a 5‐year rolling window (≈1260 trading days).
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    window = 5 * 252
    z = norm.ppf(1 - var_level)

    var = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(log_ret)):
        date = log_ret.index[i]
        data = log_ret.iloc[i - window : i]
        mu = data.mean()
        sigma = data.std()
        q = 5 * mu + z * np.sqrt(5) * sigma
        S = prices.loc[date]
        loss = -S * (np.exp(q) - 1)
        var.loc[date] = max(loss, 0.0)
    return var.dropna()


def compute_es(prices: pd.Series, es_level: float) -> pd.Series:
    """
    5-day parametric ES at es_level using GBM parameters estimated
    over a 5‐year rolling window (≈1260 days), closed-form.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    window = 5 * 252
    alpha = 1 - es_level
    z_alpha = norm.ppf(alpha)

    es = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(log_ret)):
        date = log_ret.index[i]
        data  = log_ret.iloc[i - window : i]
        mu    = data.mean()
        sigma = data.std()
        mu5   = 5 * mu
        sig5  = np.sqrt(5) * sigma

        # VaR quantile in log-return space
        # q = mu5 + sig5 * z_alpha

        # conditional moment: E[e^{R_5} | R_5 <= q]
        # = exp(mu5 + 0.5*sig5^2) * Phi(z_alpha - sig5) / Phi(z_alpha)
        phi_tail = norm.cdf(z_alpha)
        cond_moment = np.exp(mu5 + 0.5 * sig5**2) * norm.cdf(z_alpha - sig5) / phi_tail

        S = prices.loc[date]
        # dollar ES = E[ loss ] = S * (1 - E[e^{R_5} | tail])
        es.loc[date] = S * (1 - cond_moment)

    return es.dropna()
