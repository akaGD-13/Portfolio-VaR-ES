import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

import parametric5yr
import parametric_ewm
import historical
import montecarlo

def prompt_file():
    while True:
        path = input("Enter the relative path to CSV file (dates as index, security codes as columns): (software/data/portfolio.csv)").strip()
        if os.path.isfile(path):
            return path
        print(f"File not found: {path}")

def prompt_confidence(name):
    while True:
        s = input(f"Enter {name} confidence level (decimal between 0 and 1): (0.99)").strip()
        try:
            val = float(s)
            if 0 < val < 1:
                return val
            print("Value must be between 0 and 1.")
        except ValueError:
            print("Invalid number, please try again.")

def main():
    price_file = prompt_file()
    var_level  = prompt_confidence("VaR")
    es_level   = prompt_confidence("ES")

    # Load CSV
    try:
        df = pd.read_csv(price_file, parse_dates=True, index_col=0)
    except Exception as e:
        print(f"Error reading {price_file}: {e}", file=sys.stderr)
        sys.exit(1)
    if df.empty:
        print("Error: price file is empty", file=sys.stderr)
        sys.exit(1)

    # Compute portfolio value as sum of all stock columns
    portfolio = df.sum(axis=1).dropna()

    # Fixed parameters
    LAMBDA     = 0.9989
    WINDOW     = 5 * 252
    N_SIMS     = 10000

    # Compute VaR and ES series
    var1 = parametric5yr.compute_var(portfolio, var_level)
    es1  = parametric5yr.compute_es(portfolio, es_level)
    var2 = parametric_ewm.compute_var(portfolio, var_level, LAMBDA)
    # es2  = parametric_ewm.compute_es(portfolio, es_level, LAMBDA)
    var3 = historical.compute_var(portfolio, var_level, WINDOW)
    es3  = historical.compute_es(portfolio, es_level, WINDOW)
    var4 = montecarlo.compute_var(portfolio, var_level, WINDOW, N_SIMS)
    es4  = montecarlo.compute_es(portfolio, es_level, WINDOW, N_SIMS)

    # Plot VaR comparison
    plt.figure(figsize=(10,6))
    for series, label in [
        (var1, "Parametric 5yr"),
        (var2, "Parametric EWM"),
        (var3, "Historical"),
        (var4, "Monte Carlo")
    ]:
        plt.plot(series.index, series, label=label)
    plt.title(f"5-day VaR @ {var_level*100:.1f}%")
    plt.legend()
    plt.savefig("output/var_comparison.png", dpi=150)
    plt.close()

    # Plot ES comparison
    plt.figure(figsize=(10,6))
    for series, label in [
        (es1, "Parametric 5yr"),
        # (es2, "Parametric EWM"),
        (es3, "Historical"),
        (es4, "Monte Carlo")
    ]:
        plt.plot(series.index, series, label=label)
    plt.title(f"5-day ES @ {es_level*100:.1f}%")
    plt.legend()
    plt.savefig("output/es_comparison.png", dpi=150)
    plt.close()

    # Print summary
    print(f"{'Method':<15}{'Latest VaR':>12}{'Latest ES':>12}")
    for name, v, e in [
        ("Parametric5yr", var1, es1),
        # ("ParametricEWM", var2, es2),
        ("Historical",    var3, es3),
        ("MonteCarlo",    var4, es4)
    ]:
        print(f"{name:<15}{v.iloc[-1]:12.2f}{e.iloc[-1]:12.2f}")

if __name__ == "__main__":
    main()
