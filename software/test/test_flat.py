import sys, os
import pandas as pd
import numpy as np

# allow imports from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import parametric5yr
import parametric_ewm
import historical
import montecarlo


def test_flat_prices():
    # 5 years of flat prices
    dates = pd.bdate_range('2020-01-01', periods=5*252)
    prices = pd.Series(100.0, index=dates)

    # parametric5yr: VaR & ES zero
    v1 = parametric5yr.compute_var(prices, 0.99)
    e1 = parametric5yr.compute_es(prices, 0.975)
    assert np.allclose(v1, 0)
    assert np.allclose(e1, 0)

    # parametric_ewm: only VaR zero
    v2 = parametric_ewm.compute_var(prices, 0.99, lambda_=0.9989)
    assert np.allclose(v2, 0)

    # historical: VaR & ES zero
    v3 = historical.compute_var(prices, 0.99, window_days=5*252)
    e3 = historical.compute_es(prices, 0.975, window_days=5*252)
    assert np.allclose(v3, 0)
    assert np.allclose(e3, 0)

    # montecarlo: VaR & ES zero (use smaller sims for speed)
    v4 = montecarlo.compute_var(prices, 0.99, window_days=5*252, n_sims=1_000)
    e4 = montecarlo.compute_es(prices, 0.975, window_days=5*252, n_sims=1_000)
    assert np.allclose(v4, 0)
    assert np.allclose(e4, 0)
