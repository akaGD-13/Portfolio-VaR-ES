# Test Results

**Date executed:** May 13, 2025

---

## 1. Test Suite (1-4) Execution

```bash
$ pytest software/test -q

....
4 passed in 1.38s
````

All four core tests passed successfully.

1. **Flat-Price Unit Test**  
   → All VaR/ES series = 0.  

2. **Parametric Closed-Form**  
   → Last-point VaR/ES matched theory to 1e-12 tolerance.  

3. **Backtest Exception Frequency**  
   → Observed frequency 0.0094 vs. nominal 0.01 (within ±0.005).  

4. **Monte Carlo vs. Parametric**  
   → Perfect series equality on σ=0 path.  

5. **Portfolio Consistency Visualization**  
   - **VaR @99%** and **ES @99%** plots (see `var_comparison.png`, `es_comparison.png`)  
   - No sustained method deviated by more than ±20% of the median curve.  


---

## 2. Visual Comparison Plots

### 2.1 5-day ES @ 99.0% Comparison

![ES Comparison](es_comparison.png)

*No method shows a sustained, drastic deviation beyond ±20% of the group median.*

### 2.2 5-day VaR @ 99.0% Comparison

![VaR Comparison](var_comparison.png)

*The four VaR curves remain roughly aligned over the full history.*

---

