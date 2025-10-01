"""
run.py — example pipeline
NOTE: This script requires internet for Yahoo Finance unless cached data exists.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from data import load_prices, compute_returns
from signals import momentum_signal, mean_reversion_zscore, combine_signals
from backtest import backtest
from metrics import sharpe, max_drawdown, equity_curve, avg_turnover
from plots import plot_equity_curve, plot_drawdown

RUN_CONFIG = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "start": "2018-01-01",
    "end": str(date.today()),
    "momentum_lb": 126,
    "mr_lb": 20,
    "cost_bps": 10.0,
    "max_gross": 1.0,
    "out_dir": "outputs",
}

def main() -> None:
    out_dir = RUN_CONFIG["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 1) Data
    prices = load_prices(RUN_CONFIG["tickers"], RUN_CONFIG["start"], RUN_CONFIG["end"])

    # 2) Signals
    mom = momentum_signal(prices, RUN_CONFIG["momentum_lb"])
    rets = compute_returns(prices, method="simple")
    mr = mean_reversion_zscore(rets, RUN_CONFIG["mr_lb"])
    sig = combine_signals([mom, mr])

    # 3) Backtest
    bt = backtest(prices, sig, cost_bps=RUN_CONFIG["cost_bps"], max_gross=RUN_CONFIG["max_gross"])

    # 4) Metrics
    ec = equity_curve(bt["net_returns"], start_value=1.0)
    s = sharpe(bt["net_returns"])
    mdd = max_drawdown(ec)
    avg_to = avg_turnover(bt["turnover"])

    # 5) Plots
    plot_equity_curve(ec, os.path.join(out_dir, "equity_curve.png"))
    plot_drawdown(ec, os.path.join(out_dir, "drawdown.png"))

    # 6) Summary
    summary = f"""# Backtest Summary

**Period:** {RUN_CONFIG["start"]} to {RUN_CONFIG["end"]}  
**Tickers:** {", ".join(RUN_CONFIG["tickers"])}  

## Performance
- Sharpe (net): {s:.2f}
- Max Drawdown: {mdd:.2%}
- Avg Daily Turnover: {avg_to:.2f}
- Costs (bps): {RUN_CONFIG["cost_bps"]}

## Parameters
- Momentum lookback: {RUN_CONFIG["momentum_lb"]}
- Mean-reversion lookback: {RUN_CONFIG["mr_lb"]}
- Max gross leverage: {RUN_CONFIG["max_gross"]}

## Notes
- Signals lagged by 1 day to prevent look-ahead.
- Proportional transaction cost model on turnover.
- Educational project — not investment advice.
"""
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(summary)

    print("Run complete. See outputs/ for artifacts.")

if __name__ == "__main__":
    main()
