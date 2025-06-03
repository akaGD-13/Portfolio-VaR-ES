# Model Validation Report

Guangda Fei 
Columbia MAFN 

May 13, 2025

---

## Contents

1. [Executive Summary](#1-executive-summary)  
2. [Introduction](#2-introduction)  
3. [Product Description](#3-product-description)  
4. [Model Description](#4-model-description)  
5. [Validation Methodology and Scope](#5-validation-methodology-and-scope)  
6. [Validation Results](#6-validation-results)  
7. [Conclusions and Recommendations](#7-conclusions-and-recommendations)  
8. [Bibliography](#8-bibliography)  

---

## 1 Executive Summary

Risk‐management code for 5-day VaR and ES on a stock (and eventual stock+option) portfolio was reviewed.  I implement four methods—parametric (5-year window), parametric (EWM), historical and Monte Carlo—and validate:

- **Accuracy** against closed-form GBM formulas  
- **Stability** via flat‐price and backtest exception‐frequency tests  
- **Consistency** Monte Carlo vs. parametric and across methods on a real portfolio

All tests passed within their acceptance criteria.  The models reliably capture tail‐risk for equity portfolios; extensions for options require adding volatility‐surface dynamics and Greeks.

---

## 2 Introduction

- **Review scope:** VaR/ES engines `parametric5yr`, `parametric_ewm`, `historical`, `montecarlo` in `/software`
- **Intended usage:** marking and risk‐management of equity and option portfolios in a Python framework  
- **Business unit:** Model Risk / Quantitative Risk Management  
---

## 3 Product Description


A standalone risk‐calculation library that:

- Takes time-series of asset prices (stocks, later options) and a portfolio definition  
- Computes 5-day VaR and ES at user-specified confidence levels via:  
  1. Parametric GBM (rolling window)  
  2. Parametric GBM (exponential weighting)  
  3. Historical empirical  
  4. Monte Carlo GBM  

Graphs and numeric outputs are produced for reporting and backtesting.

---

## 4 Model Description

### Modeling theory / assumptions

- **Parametric VaR (5 yr window)**  
  Assume the total portfolio value \(P_t\) follows a Geometric Brownian Motion:  
  $$
  dP_t = \mu\,P_t\,dt + \sigma\,P_t\,dW_t.
  $$  
  Estimate drift $\mu$ and volatility $\sigma$ from the past five years of daily log-returns. Take the 1 % left tail of the resulting 5-day log-normal return distribution as VaR.

- **Parametric VaR (EWM “5 yr equivalent”)**  
  Same GBM assumption, but compute $\mu$ and $\sigma$ by exponential weighting of all past daily log-returns with decay factor $\lambda$ = 0.9992 (so that the effective memory ≈ 5 years). Define  
  $$
  \alpha = 1 - \lambda = 0.0008.
  $$

- **Historical VaR (5 yr window)**  
  No distributional assumptions. Collect the past five years of realized 5-day P&L, sort the losses, and take the 1 % empirical quantile.

- **Monte Carlo VaR (5 yr window)**  
  Assume GBM with $\mu,\sigma$ estimated over a 5 year window. Simulate many 5-day paths of $P_t$, compute the loss distribution, and take the 1 % worst-case loss.

- **Expected Shortfall**  
  For each of the four VaR methods above, compute ES as the average of losses exceeding the VaR threshold (e.g. the worst 1 % tail for 99 % ES).

#### Options:
- **Underlying dynamics**  
  We continue to assume each stock follows GBM under both the real and risk-neutral measures:
  $$
    dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
    \quad\text{and}\quad
    dS_t = r S_t\,dt + \sigma S_t\,dW_t^Q.
  $$
- **Volatility surface dynamics**  
  Options depend on the entire implied‐volatility surface (strike, maturity).  We model its evolution via one of:
  1. **Simple “sticky delta” shocks**: treat ATM vol changes as a time-series, and shift the entire surface by historical Δσ_ATM.  
  2. **Stochastic‐volatility model** (e.g. Heston): jointly simulate $S_t$ and instantaneous variance $v_t$ under $Q$.  
  3. **SABR or GARCH‐based local‐vol model**: calibrate to current surface and simulate vol‐of‐vol.  

- **Portfolio P&L approximation**  
  For parametric VaR/ES we use a **delta‐vega‐gamma approximation** of small changes:
  $$
    \Delta P \approx \Delta_S\,\Delta S + \tfrac12\Gamma_S(\Delta S)^2
      + \text{Vega}\,\Delta\sigma
  $$
  where $(\Delta S,\Delta\sigma)$ are joint‐normally distributed with parameters estimated from historical windows or EWM.

- **Historical and Monte Carlo**  
  - **Historical**: collect five years of **total option P&L** at 5-day horizons, sorting and taking the 1% tail.  Implied vols and underlying returns enter implicitly via realized option price moves.  
  - **Monte Carlo**: simulate underlying paths (and vol paths if using Heston/SABR), reprice each option at $t+5$ with the appropriate pricing model (Black-Scholes or chosen SV model), compute P&L.

---

### Mathematical description

1. **Daily log-returns**  
   $$
     r_{t} = \ln\bigl(P_{t}/P_{t-1}\bigr).
   $$

2. **Parametric VaR**  
   - Estimate  
     $$
       \hat\mu = \frac{1}{N}\sum_{i=1}^N r_i,
       \quad
       \hat\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (r_i-\hat\mu)^2}
     $$  
     either by rolling window or EWM.  
   - The 5-day log-return $R_5\sim N(5\hat\mu,\;5\hat\sigma^2)$.  
   - The 99 % VaR is  
    $$
        \mathrm{VaR}_{0.99}
        = -\,P_t\bigl(e^{q_{0.01}} - 1\bigr),
        \quad
        q_{0.01} = 5\,\hat\mu + z_{0.01}\,\sqrt{5}\,\hat\sigma,
        \quad
        z_{0.01} = \Phi^{-1}(0.01).
    $$

3. **Historical VaR**  
   - Compute actual 5-day P&L: $\Delta P_i = P_{t_i+5} - P_{t_i}$.  
   - Sort the losses $-\Delta P$ and take its 1 % quantile.

4. **Monte Carlo VaR**  
   - Draw $M$ samples $R_5^{(j)}\sim N(5\hat\mu,5\hat\sigma^2)$.  
   - Simulate $P_t^{(j)} = P_t\,e^{R_5^{(j)}}$.  
   - Losses $\ell^{(j)} = P_t - P_t^{(j)}$.  
   - VaR = 1 % quantile of $\{\ell^{(j)}\}$.

5. **Expected Shortfall**  
   - Compute the mean of the worst 1 % of losses for 99 % ES (or worst 2.5 % for 97.5 % ES).

#### Options:
1. **Greeks**  
   - $\Delta = \partial P/\partial S,\;\Gamma = \partial^2P/\partial S^2,\;\text{Vega} = \partial P/\partial\sigma.$  
   - Compute via closed‐form formulas (Black-Scholes) or finite‐difference on numerically‐calibrated surfaces.

2. **Joint distribution**  
   - Estimate daily covariance matrix of $(\Delta S/S,\,\Delta\sigma)$ over your chosen window.  
   - Scale to 5-day horizon:  
    $$
        \mathrm{Cov}\!\Bigl(\frac{\Delta S}{S},\,\Delta\sigma\Bigr)_{5\text{-day}}
        = 5 \times 
        \mathrm{Cov}\!\Bigl(\frac{\Delta S}{S},\,\Delta\sigma\Bigr)_{\mathrm{daily}}.
    $$


3. **Parametric VaR**  
   - For each date $t$, approximate the 5-day P&L distribution as normal with mean $\mu_P$ and variance $\sigma_P^2$ given by the Greek‐weighted covariance:
    $$
       \mu_P = \Delta\,\mu_S + \text{Vega}\,\mu_\sigma + \tfrac12\,\Gamma\,\sigma_S^2,
    $$
    $$
        \sigma_P^2
        = \Delta^2\,\sigma_S^2
        \;+\;\text{Vega}^2\,\sigma_\sigma^2
        \;+\;2\,\Delta\,\text{Vega}\,\mathrm{Cov}\bigl(\Delta S,\Delta\sigma\bigr)
        \;+\;\tfrac{1}{4}\,\Gamma^2\,\mathrm{Var}\bigl((\Delta S)^2\bigr).
    $$
   - $VaR_{0.99}$ is given by the 1% quantile of the normal distribution:
        $$
        \mathrm{VaR}_{0.99}
        = \mu_P \;+\;\sigma_P\,\Phi^{-1}(0.01),
        $$
        where $\Phi^{-1}$ is the inverse CDF of the standard normal.

4. **Historical VaR / ES**  
   - Build the series of 5-day option P&L directly (price$_{t+5}$ − price$_t$) and take empirical quantiles or tail‐averages.

5. **Monte Carlo VaR / ES**  
   - Simulate scenarios $(S_{t+5},\sigma_{t+5})$, reprice options via $P(S_{t+5},\sigma_{t+5})$, compute P&L, and extract the 1% and ES statistics.

---

### Model implementation & numerical methods

- **Parameter estimation**  
  - Rolling: sample mean & std over $N$ days.  
  - EWM: use `.ewm(alpha=alpha, adjust=False)` for weighted mean & var.

- **VaR calculation**  
  - Parametric: closed-form log-normal quantile.  
  - Historical: sort & index empirical distribution.  
  - Monte Carlo: vectorized simulation via `numpy.random.normal`.

- **Backtesting**  
  - Compare realized 5-day P&L to previous-day VaR and count exceptions.

- **Greek calculation**  
  - Use Black-Scholes formulas for $\Delta,\Gamma,\text{Vega}$ with current implied vol.  
  - If using a full SV model, compute via adjoint or finite‐difference.

- **Volatility estimation**  
  - For parametric VaR: estimate daily $\mu_\sigma$, $\sigma_\sigma$, and $\mathrm{Cov}(\Delta S, \Delta\sigma)$ via rolling or EWM.

  - For Monte Carlo: calibrate SV parameters (e.g. Heston $\kappa,\theta,\xi,\rho$) to the implied surface.

- **Option repricing**  
  - Historical: simply read back historical option prices from the data source.  
  - Monte Carlo: vectorized pricing routines for large scenario sets (NumPy, Numba, or C extensions).


---

### Calibration methodology

- **Window lengths**  
  - 5 years ≈ 1 260 trading days.

- **Decay factor**  
  $$
  \lambda = 0.9992,\quad
  \alpha = 1 - \lambda = 0.0002.
  $$

- **Monte Carlo sample size**  
  - $M = 10{,}000$ paths to stabilize tail estimates.

#### Options:
- **Window lengths**  
  - Use the same 5-year windows for underlying and vol estimates.  
- **SV calibration**  
  - Fit model to current surface daily, update parameters with EWM.


---

### Model usage

* The top‐level **main.py** acts as a launcher, prompting the user to choose between:

  1. **historical\_calibration.py** — loads CSV, calibrates μ/σ over historical window, then runs all VaR/ES methods
  2. **input\_mu\_sigma.py**      — prompts user for manual μ/σ, then runs parametric VaR & ES only
* There are **four VaR engines**:

  * `parametric5yr` (rolling‐window GBM)
  * `parametric_ewm` (exponentially weighted GBM, VaR only)
  * `historical` (empirical P\&L)
  * `montecarlo` (GBM Monte Carlo)
* There are **three ES engines** (all except `parametric_ewm`).

---

### Model exposure

- Captures market risk of the portfolio $P$.  
- Supports both long- and short-position loss profiles.  
- Future extensions: multi-asset portfolios, time-varying correlation, jumps.

---

## 5 Validation Methodology and Scope

- (details in test plan)

We exercised a five-point test plan:

1. **Flat-Price Unit Test**  
   - Constant prices → zero VaR/ES for all methods (ES skipped for EWM).  
2. **Parametric Closed-Form Consistency**  
   - Deterministic GBM (σ=0) → exact match to analytic VaR/ES.  
3. **Backtest Exception Frequency**  
   - 10-yr GBM path → realized 5-day exception frequency within ±0.005 of nominal 1%.  
4. **Monte Carlo vs. Parametric Agreement**  
   - σ=0 path → parametric and MC VaR series are identical.  
5. **Portfolio Consistency Visualization**  
   - Real multi-stock portfolio → overlaid VaR and ES time series showed no method drifting by more than ±20% relative to the group median.

First 4 tests were automated with `pytest` and simple plotting scripts, covering both unit‐level and end-to-end validation.

---

## 6 Validation Results

- details in test result
- all tests passed

---

## 7 Conclusions and Recommendations

The four VaR/ES engines:

- **Pass all unit and end-to-end tests**, demonstrating correct implementation of GBM assumptions.  
- **Agree closely** on real-data portfolio risk metrics, ensuring robustness.  

**Limitations & next steps:**

- Current scope: **stocks only**.  Options require adding volatility‐surface dynamics.  
- Recommendation: implement Greeks-based parametric and full SV Monte Carlo for options, then re-run the validation suite with option P&L data.  
- Investigate **non-GBM features** (jumps, fat tails) if backtests on extreme events show undercoverage.

---

## 8 Bibliography

- [1] Stein, H. J. “Model Validation Report Template.” Strategic Risk Research Group, November 18 2015.   
