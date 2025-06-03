import numpy as np
import pandas as pd
from scipy.stats import norm
from math import exp, sqrt

# reuse bs_price from parametric file or re-import here
from option_parametric import bs_price

def compute_var(S, K, T, mu, sigma, position, var_level, r=0.05, q=0.0,
                option_type='call', n_sims=10000) -> float:
    """
    Monte Carlo VaR for an option position.
    """
    # simulate 5-day underlying
    days = 5
    dt = days * (1/252)
    drift = (mu - 0.5*sigma**2) * dt
    vol = sigma * sqrt(dt)
    Z = np.random.randn(n_sims)
    S5 = S * np.exp(drift + vol * Z)

    # reprice options
    P0 = bs_price(S, K, r, q, T, sigma, option_type)
    P5 = np.array([bs_price(s5, K, r, q, T-dt, sigma, option_type) for s5 in S5])

    losses = (P0 - P5) * position
    var = np.percentile(losses, 100*(1-var_level))
    return max(var, 0.0)


def compute_es(S, K, T, mu, sigma, position, es_level, r=0.05, q=0.0,
               option_type='call', n_sims=10000) -> float:
    """
    Monte Carlo ES for an option position.
    """
    days = 5
    dt = days * (1/252)
    drift = (mu - 0.5*sigma**2) * dt
    vol = sigma * sqrt(dt)
    Z = np.random.randn(n_sims)
    S5 = S * np.exp(drift + vol * Z)

    P0 = bs_price(S, K, r, q, T, sigma, option_type)
    P5 = np.array([bs_price(s5, K, r, q, T-dt, sigma, option_type) for s5 in S5])

    losses = (P0 - P5) * position
    cutoff = np.percentile(losses, 100*(1-es_level))
    tail = losses[losses >= cutoff]
    es = tail.mean() if len(tail)>0 else 0.0
    return max(es, 0.0)

def compute_var_series(prices: pd.Series, K: float, T: float,
                       var_level: float, window_days: int,
                       position: float, r=0.05, q=0.0,
                       option_type='call', n_sims=10000) -> pd.Series:
    """Rolling Monte Carlo VaR series for an option."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    var_ser = pd.Series(index=prices.index, dtype=float)
    for i in range(window_days, len(log_ret)):
        date = log_ret.index[i]
        window_data = log_ret.iloc[i-window_days:i]
        sigma_est = window_data.std()*np.sqrt(252)
        mu = window_data.mean()*(252) + 0.5*sigma_est**2
        var_ser.loc[date] = compute_var(
            prices.loc[date], K, T, mu, sigma_est,
            position, var_level, r, q, option_type, n_sims
        )
    return var_ser.dropna()


def compute_es_series(prices: pd.Series, K: float, T: float,
                      es_level: float, window_days: int,
                      position: float, r=0.05, q=0.0,
                      option_type='call', n_sims=10000) -> pd.Series:
    """Rolling Monte Carlo ES series for an option."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    es_ser = pd.Series(index=prices.index, dtype=float)
    for i in range(window_days, len(log_ret)):
        date = log_ret.index[i]
        window_data = log_ret.iloc[i-window_days:i]
        sigma_est = window_data.std()*np.sqrt(252)
        mu = window_data.mean()*252 + 0.5*sigma_est**2
        es_ser.loc[date] = compute_es(
            prices.loc[date], K, T, mu, sigma_est,
            position, es_level, r, q,
            option_type, n_sims
        )
    return es_ser.dropna()