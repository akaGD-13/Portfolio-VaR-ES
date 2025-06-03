import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import parametric5yr

def simulate_gbm(mu, sigma, S0=100.0, days=10*252, seed=1):
    np.random.seed(seed)
    dt = 1/252
    increments = np.random.normal(
        (mu - 0.5*sigma**2)*dt,
        sigma*np.sqrt(dt),
        size=days
    ).cumsum()
    return pd.Series(S0 * np.exp(increments), index=pd.RangeIndex(days))

def test_exception_frequency():
    # low-vol GBM for stable exception rate
    mu, sigma = 0.00, 0.01
    prices = simulate_gbm(mu, sigma)
    var_series = parametric5yr.compute_var(prices, 0.99)

    # realized 5-day P&L
    pnl5 = (prices.shift(-5) - prices).loc[var_series.index]
    exceptions = (pnl5 < -var_series).sum()
    freq = exceptions / len(var_series)

    # expect ~1% ± 0.5%
    assert abs(freq - 0.01) < 0.005
