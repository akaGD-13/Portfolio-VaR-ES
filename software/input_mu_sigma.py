# portfolio_var_es.py

import numpy as np
from scipy.stats import norm

def parametric_var(S, pos, mu, sigma, var_level):
    """
    5-day parametric VaR for a stock position.
    """
    # 5-day drift & vol
    mu5  = (mu - 0.5*sigma**2) * 5
    sig5 = sigma * np.sqrt(5)
    z    = norm.ppf(1 - var_level)
    q    = mu5 + z*sig5
    loss = -S*(np.exp(q) - 1)*pos
    return loss

def parametric_es(S, pos, mu, sigma, es_level):
    """
    5-day parametric ES for a stock via closed-form tail expectation.
    """
    # 1) compute 5-day drift & vol
    mu5  = (mu - 0.5*sigma**2) * 5
    sig5 = sigma * np.sqrt(5)

    # 2) find the tail‐quantile point z_alpha
    tail_prob = 1 - es_level          # e.g. 0.025 for 97.5% ES
    z_alpha   = norm.ppf(tail_prob)   # ≈ -1.96

    # 3) conditional moment E[e^{R5} | R5 <= q]
    #    = exp(mu5 + 0.5*sig5^2) * Phi(z_alpha - sig5) / tail_prob
    numer = np.exp(mu5 + 0.5*sig5**2) * norm.cdf(z_alpha - sig5)
    cond_moment = numer / tail_prob

    # 4) dollar‐loss ES: S * (1 - E[e^{R5}|tail]) * position
    es_loss = -S * (cond_moment - 1) * pos
    return es_loss

def mc_var(S, pos, mu, sigma, var_level, n_sims):
    """
    5-day Monte Carlo VaR for a stock position.
    """
    mu5  = (mu - 0.5*sigma**2) * 5
    sig5 = sigma * np.sqrt(5)
    sims = np.random.normal(mu5, sig5, size=n_sims)
    losses = -S*(np.exp(sims) - 1)*pos
    return np.percentile(losses, 100*var_level)

def mc_es(S, pos, mu, sigma, es_level, n_sims):
    """
    5-day Monte Carlo ES for a stock position.
    """
    mu5  = (mu - 0.5*sigma**2) * 5
    sig5 = sigma * np.sqrt(5)
    sims = np.random.normal(mu5, sig5, size=n_sims)
    losses = -S*(np.exp(sims) - 1)*pos
    cutoff = np.percentile(losses, 100*es_level)
    tail   = losses[losses >= cutoff]
    return tail.mean() if len(tail)>0 else 0.0

# Option pricing and Greeks
from math import log, sqrt, exp
def bs_price_delta(S, K, r, T, sigma, q=0.0, option_type='call'):
    """
    Black-Scholes price & delta for European option.
    Returns (price, delta).
    """
    d1 = (log(S/K)+(r - q + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if option_type=='call':
        price = S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        delta = exp(-q*T)*norm.cdf(d1)
    else:
        price = K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)
        delta = exp(-q*T)*(norm.cdf(d1)-1)
    return price, delta

def option_parametric_var(S, pos, K, T, r, q, mu, sigma, var_level, option_type='call'):
    """
    5-day parametric VaR for an option via delta-normal.
    """
    P0, delta = bs_price_delta(S, K, r, T, sigma, q, option_type)
    # treat ΔP ≈ delta * ΔS
    return parametric_var(S, delta*pos, mu, sigma, var_level)

def option_parametric_es(S, pos, K, T, r, q, mu, sigma, es_level, option_type='call'):
    """
    5-day parametric ES for an option via delta-normal.
    """
    P0, delta = bs_price_delta(S, K, r, T, sigma, q, option_type)
    return parametric_es(S, delta*pos, mu, sigma, es_level)

def option_mc_var(S, pos, K, T, r, q, mu, sigma, var_level, n_sims, option_type='call'):
    """
    5-day MC VaR for an option by full repricing.
    """
    mu5  = (mu - 0.5*sigma**2) * 5
    sig5 = sigma * np.sqrt(5)
    sims = np.random.normal(mu5, sig5, size=n_sims)
    S5   = S * np.exp(sims)
    losses = []
    for s5 in S5:
        P0, _ = bs_price_delta(S, K, r, T, sigma, q, option_type)
        P5, _ = bs_price_delta(s5, K, r, T-5/252, sigma, q, option_type)
        losses.append((P0 - P5)*pos)
    return np.percentile(losses, 100*var_level)

def option_mc_es(S, pos, K, T, r, q, mu, sigma, es_level, n_sims, option_type='call'):
    """
    5-day MC ES for an option by repricing.
    """
    mu5  = (mu - 0.5*sigma**2) * 5
    sig5 = sigma * np.sqrt(5)
    sims = np.random.normal(mu5, sig5, size=n_sims)
    losses = []
    for s5 in sims:
        S5 = S * np.exp(s5)
        P0, _ = bs_price_delta(S, K, r, T, sigma, q, option_type)
        P5, _ = bs_price_delta(S5, K, r, T-5/252, sigma, q, option_type)
        losses.append((P0 - P5)*pos)
    losses = np.array(losses)
    cutoff = np.percentile(losses, 100*es_level)
    tail = losses[losses >= cutoff]
    return tail.mean() if len(tail)>0 else 0.0

def main():
    import sys

    # Prompt user
    n = int(input("Number of stocks: "))
    stocks = []
    for i in range(n):
        code    = input(f"Stock #{i+1} code: ")
        pos     = float(input(f"  Position (# shares): "))
        S       = float(input(f"  Current price of {code}: "))
        mu      = float(input("  Drift (mu): "))
        sigma   = float(input("  Volatility (sigma): "))
        stocks.append((code, pos, S, mu, sigma))

    m = int(input("Number of options: "))
    options = []
    for j in range(m):
        code    = input(f"Option #{j+1} underlying code: ")
        pos     = float(input(f"  Position (# contracts): "))
        S       = float(input(f"  Current price of {code}: "))
        K       = float(input("  Strike: "))
        T       = float(input("  Time to maturity (yrs): "))
        mu      = float(input("  Drift (mu): "))
        sigma   = float(input("  Volatility (sigma): "))
        r       = float(input("  Risk-free rate (r) [0.05]: ") or 0.05)
        q       = float(input("  Dividend yield (q) [0.0]: ") or 0.0)
        otype   = input("  Option type (call/put) [call]: ") or "call"
        options.append((code, pos, S, K, T, r, q, mu, sigma, otype))

    var_level = float(input("VaR confidence (e.g. 0.99): "))
    es_level  = float(input("ES confidence (e.g. 0.975): "))
    n_sims    = int(input("MC sims (e.g. 10000): "))

    print("\n=== Stock Parametric VaR/ES ===")
    for code, pos, S, mu, sigma in stocks:
        v_p = parametric_var(S, pos, mu, sigma, var_level)
        e_p = parametric_es(S, pos, mu, sigma, es_level)
        v_m = mc_var(S, pos, mu, sigma, var_level, n_sims)
        e_m = mc_es(S, pos, mu, sigma, es_level, n_sims)
        print(f"{code}:  Parametric VaR={v_p:.2f}, ES={e_p:.2f} | MC VaR={v_m:.2f}, ES={e_m:.2f}")

    print("\n=== Option Parametric VaR/ES ===")
    for code, pos, S, K, T, r, q, mu, sigma, otype in options:
        v_p = option_parametric_var(S, pos, K, T, r, q, mu, sigma, var_level, otype)
        e_p = option_parametric_es(S, pos, K, T, r, q, mu, sigma, es_level, otype)
        v_m = option_mc_var(S, pos, K, T, r, q, mu, sigma, var_level, n_sims, otype)
        e_m = option_mc_es(S, pos, K, T, r, q, mu, sigma, es_level, n_sims, otype)
        print(f"{code}:  Parametric VaR={v_p:.2f}, ES={e_p:.2f} | MC VaR={v_m:.2f}, ES={e_m:.2f}")


if __name__ == "__main__":
    main()
    