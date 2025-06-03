import sys, os
import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm

# allow import of your modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import parametric5yr

def simulate_det_gbm(mu, sigma, S0=100.0, days=6*252):
    """Deterministic GBM with zero volatility (sigma=0)."""
    dt = 1/252
    inc = (mu - 0.5 * sigma**2) * dt
    times = np.arange(1, days+1)
    logprice = np.log(S0) + inc * times
    return pd.Series(np.exp(logprice), index=pd.RangeIndex(days))

def test_parametric_closed_form():
    mu, sigma = 0.05, 0.0       # zero volatility case
    prices    = simulate_det_gbm(mu, sigma)
    var_level = 0.99
    es_level  = 0.975

    # compute daily drift (what code estimates)
    drift_log_daily = (mu - 0.5 * sigma**2) / 252  # = 0.05/252

    # scale to 5-day horizon like in the code
    mu5   = 5 * drift_log_daily
    sig5  = np.sqrt(5) * (sigma / np.sqrt(252))   # = 0 when sigma=0

    alpha = 1 - es_level
    z_var = norm.ppf(1 - var_level)
    z_es  = norm.ppf(alpha)

    S_t = prices.iloc[-1]

    # theoretical VaR (clamped at zero)
    theo_loss = S_t * (1 - np.exp(5 * drift_log_daily + z_var * sig5))
    theo_var  = max(theo_loss, 0.0)

    # theoretical ES using closed‐form tail expectation
    tail_prob   = norm.cdf(z_es)
    cond_expect = (
        np.exp(mu5 + 0.5 * sig5**2)
        * norm.cdf(z_es - sig5)
        / tail_prob
    )
    theo_es = S_t * (1 - cond_expect)

    v = parametric5yr.compute_var(prices, var_level).iloc[-1]
    e = parametric5yr.compute_es(prices, es_level).iloc[-1]

    assert v == pytest.approx(theo_var, rel=1e-12)
    assert e == pytest.approx(theo_es,   rel=1e-12)
