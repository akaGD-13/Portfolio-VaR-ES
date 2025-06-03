import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import parametric5yr
import montecarlo

def simulate_gbm(mu, sigma, S0=100.0, days=8*252, seed=2):
    np.random.seed(seed)
    dt = 1/252
    increments = np.random.normal(
        (mu - 0.5*sigma**2)*dt,
        sigma*np.sqrt(dt),
        size=days
    ).cumsum()
    return pd.Series(S0 * np.exp(increments), index=pd.RangeIndex(days))

def test_mc_agreement():
    # same GBM path for both
    mu, sigma = 0.05, 0.20
    prices = simulate_gbm(mu, sigma)
    var_level = 0.99
    window    = 5 * 252

    v_param = parametric5yr.compute_var(prices, var_level)
    v_mc    = montecarlo.compute_var(prices, var_level, window, n_sims=20_000)

    # align
    idx = v_param.index.intersection(v_mc.index)
    diff = (v_mc.loc[idx] - v_param.loc[idx]).abs()
    rel  = diff / v_param.loc[idx]

    # mean relative difference < 10% (allow some MC noise)
    assert rel.mean() < 0.10
