import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import norm
from math import exp, sqrt

def bs_price(S, K, r, q, T, sigma, option_type='call'):
    """
    Black-Scholes price for European call or put.
    """
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
    return price


def compute_var(S, K, T, mu, sigma, position, var_level, r=0.05, q=0.0, option_type='call') -> float:
    """
    Parametric VaR for an option position via delta-normal approx.
    Parameters:
      S, K: spot price and strike
      T    : time to maturity (in years)
      mu   : drift of underlying
      sigma: volatility of underlying
      position: number of option contracts (positive for long)
      var_level: VaR confidence (e.g. 0.99)
      r    : risk-free rate (default 0.05)
      q    : dividend yield (default 0)
      option_type: 'call' or 'put'
    Returns:
      VaR (positive number)
    """
    # compute option delta
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    delta = np.exp(-q*T) * norm.cdf(d1) if option_type=='call' else np.exp(-q*T)*(norm.cdf(d1)-1)

    # 5-day scaling
    days = 5
    daily_mu = mu - 0.5*sigma**2
    drift = daily_mu * days
    vol5 = sigma * sqrt(days)

    # portfolio P&L approx mean and std
    mu_P = delta * S * (exp(drift)-1)
    sigma_P = abs(delta * S * (exp(drift) * sqrt(np.exp(vol5**2)-1)))  # approx

    z = norm.ppf(1 - var_level)
    var = -(mu_P + z * sigma_P) * position
    return max(var, 0.0)


def compute_es(S, K, T, mu, sigma, position, es_level, r=0.05, q=0.0, option_type='call') -> float:
    """
    Parametric ES for an option position via normal tail formula.
    Returns ES (positive number).
    """
    # compute delta same as above
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    delta = np.exp(-q*T) * norm.cdf(d1) if option_type=='call' else np.exp(-q*T)*(norm.cdf(d1)-1)

    days = 5
    daily_mu = mu - 0.5*sigma**2
    drift = daily_mu * days
    vol5 = sigma * sqrt(days)

    mu_P = delta * S * (exp(drift)-1)
    sigma_P = abs(delta * S * (exp(drift) * sqrt(np.exp(vol5**2)-1)))

    alpha = 1 - es_level
    z = norm.ppf(alpha)
    phi = norm.pdf(z)
    es = -(mu_P + sigma_P * phi/alpha) * position
    return max(es, 0.0)

def compute_var_series(prices: pd.Series, K: float, T: float,
                       var_level: float, window_days: int,
                       position: float, r=0.05, q=0.0,
                       option_type='call') -> pd.Series:
    """
    Rolling 5-day parametric VaR series for an option.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    var_ser = pd.Series(index=prices.index, dtype=float)
    for i in range(window_days, len(log_ret)):
        date = log_ret.index[i]
        window_prices = prices.iloc[i-window_days:i]
        sigma_est = np.log(window_prices / window_prices.shift(1)).dropna().std() / np.sqrt(1/252)
        mu = np.log(window_prices / window_prices.shift(1)).dropna().mean() / (1/252) + 0.5* sigma_est**2
        var_ser.loc[date] = compute_var(
            prices.loc[date], K, T, mu, sigma_est,
            position, var_level, r, q, option_type
        )
    return var_ser.dropna()


def compute_es_series(prices: pd.Series, K: float, T: float,
                      es_level: float, window_days: int,
                      position: float, r=0.05, q=0.0,
                      option_type='call') -> pd.Series:
    """
    Rolling 5-day parametric ES series for an option.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    es_ser = pd.Series(index=prices.index, dtype=float)
    for i in range(window_days, len(log_ret)):
        date = log_ret.index[i]
        window_prices = prices.iloc[i-window_days:i]
        sigma_est = np.log(window_prices / window_prices.shift(1)).dropna().std() / np.sqrt(1/252)
        mu = np.log(window_prices / window_prices.shift(1)).dropna().mean() / (1/252) + 0.5* sigma_est**2
        es_ser.loc[date] = compute_es(
            prices.loc[date], K, T, mu, sigma_est,
            position, es_level, r, q, option_type
        )
    return es_ser.dropna()