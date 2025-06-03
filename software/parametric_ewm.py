import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_var(prices: pd.Series, var_level: float, lambda_: float) -> pd.Series:
    """
    5-day VaR at var_level using GBM parameters estimated
    by exponential weighting (decay lambda_).
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    alpha = 1 - lambda_
    mu_ewm = log_ret.ewm(alpha=alpha, adjust=False).mean()
    var_ewm = log_ret.ewm(alpha=alpha, adjust=False).var()
    sigma_ewm = np.sqrt(var_ewm)
    z = norm.ppf(1 - var_level)

    var = pd.Series(index=prices.index, dtype=float)
    for date in log_ret.index:
        mu = mu_ewm.loc[date]
        sigma = sigma_ewm.loc[date]
        if pd.isna(mu) or pd.isna(sigma):
            continue
        q = 5 * mu + z * np.sqrt(5) * sigma
        S = prices.loc[date]
        loss = -S * (np.exp(q) - 1)
        var.loc[date] = max(loss, 0.0)
    return var.dropna()

def compute_es(prices: pd.Series, es_level: float, lambda_: float, n_sims: int = 10000) -> pd.Series:
    """
    5-day ES at es_level using GBM parameters estimated
    by exponential weighting (decay lambda_). Uses Monte Carlo.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    alpha = 1 - lambda_
    mu_ewm = log_ret.ewm(alpha=alpha, adjust=False).mean()
    var_ewm = log_ret.ewm(alpha=alpha, adjust=False).var()
    sigma_ewm = np.sqrt(var_ewm)
    tail = 1 - es_level

    es = pd.Series(index=prices.index, dtype=float)
    for date in log_ret.index:
        mu = mu_ewm.loc[date]
        sigma = sigma_ewm.loc[date]
        if pd.isna(mu) or pd.isna(sigma):
            continue
        mean5 = 5 * mu
        std5 = np.sqrt(5) * sigma
        sims = np.random.normal(mean5, std5, size=n_sims)
        S = prices.loc[date]
        losses = -S * (np.exp(sims) - 1)
        cutoff = np.percentile(losses, 100 * tail)
        es.loc[date] = losses[losses >= cutoff].mean()
    return es.dropna()
