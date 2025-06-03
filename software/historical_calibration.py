# main.py

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
        path = input(
            "Enter the relative path to CSV file "
            "(dates as index, security codes as columns)\n"
            " [e.g. software/data/portfolio.csv]: "
        ).strip()
        if os.path.isfile(path):
            return path
        print(f"File not found: {path}")

def prompt_confidence(name):
    while True:
        s = input(f"Enter {name} confidence level (decimal between 0 and 1) [e.g. 0.99]: ").strip()
        try:
            val = float(s)
            if 0 < val < 1:
                return val
            print("Value must be between 0 and 1.")
        except ValueError:
            print("Invalid number, please try again.")

def prompt_stock_positions():
    n = int(input("Enter number of stock positions: ").strip())
    stocks = []
    for i in range(1, n+1):
        code = input(f"  Code for stock #{i}: ").strip()
        pos  = float(input(f"  Shares for {code}: ").strip())
        stocks.append((code, pos))
    return stocks

def main():
    # --- load data and confidences ---
    price_file = prompt_file()
    var_level  = prompt_confidence("VaR")
    es_level   = prompt_confidence("ES")

    df = pd.read_csv(price_file, parse_dates=True, index_col=0)
    if df.empty:
        print("Error: price file is empty", file=sys.stderr)
        sys.exit(1)

    # --- build stock portfolio series ---
    stocks = prompt_stock_positions()
    stock_series = pd.Series(0.0, index=df.index)
    for code, pos in stocks:
        if code not in df.columns:
            print(f"Unknown stock code: {code}", file=sys.stderr)
            sys.exit(1)
        stock_series += df[code] * pos
    stock_series = stock_series.dropna()

    # --- compute stock-only VaR & ES across methods ---
    LAMBDA   = 0.9989
    WINDOW   = 5 * 252
    N_SIMS   = 10000

    var1 = parametric5yr.compute_var(stock_series, var_level)
    es1  = parametric5yr.compute_es(stock_series, es_level)

    var2 = parametric_ewm.compute_var(stock_series, var_level, LAMBDA)
    es2  = parametric_ewm.compute_es(stock_series, es_level, LAMBDA)

    var3 = historical.compute_var(stock_series, var_level, WINDOW)
    es3  = historical.compute_es(stock_series, es_level, WINDOW)

    var4 = montecarlo.compute_var(stock_series, var_level, WINDOW, N_SIMS)
    es4  = montecarlo.compute_es(stock_series, es_level, WINDOW, N_SIMS)

    # --- print summary of latest VaR & ES ---
    print("\nStock Portfolio VaR and ES:")
    print(f"{'Method':<20}{'VaR':>12}{'ES':>12}")
    for name, v, e in [
        ("Parametric 5yr", var1,  es1),
        ("Parametric EWM", var2,  es2),
        ("Historical",      var3,  es3),
        ("Monte Carlo",     var4,  es4),
    ]:
        print(f"{name:<20}{v.iloc[-1]:12.2f}{e.iloc[-1]:12.2f}")

    # --- plot VaR comparison ---
    os.makedirs("output", exist_ok=True)
    plt.figure(figsize=(10,6))
    for series, label in [
        (var1, "Parametric 5yr"),
        (var2, "Parametric EWM"),
        (var3, "Historical"),
        (var4, "Monte Carlo"),
    ]:
        plt.plot(series.index, series, label=label, alpha=0.8)
    plt.title(f"5-day VaR @ {var_level*100:.1f}% (Portfolio)")
    plt.ylabel("VaR (Loss)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("output/var_comparison.png", dpi=150)
    plt.close()

    # --- plot ES comparison ---
    plt.figure(figsize=(10,6))
    for series, label in [
        (es1, "Parametric 5yr"),
        # (es2, "Parametric EWM"),
        (es3, "Historical"),
        (es4, "Monte Carlo"),
    ]:
        plt.plot(series.index, series, label=label, alpha=0.8)
    plt.title(f"5-day ES @ {es_level*100:.1f}% (Portfolio)")
    plt.ylabel("ES (Loss)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("output/es_comparison.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
