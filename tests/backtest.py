from __future__ import annotations
import numpy as np
import pandas as pd

def backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: float = 10.0,
    max_gross: float = 1.0,
    lag_signal: int = 1,
) -> dict:
    # Align data
    prices, signal = prices.align(signal, join="inner")
    rets = prices.pct_change().fillna(0.0)

    # Cross-sectional standardization (per day)
    sig = signal.copy()
    sig = sig.sub(sig.mean(axis=1), axis=0)
    denom = sig.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = sig.div(denom + 1e-9, axis=0)

    # No-lookahead: lag signals
    z = z.shift(lag_signal)

    # Identify warm-up rows (all-NaN after lag)
    warmup = ~z.notna().any(axis=1)

    # Normalize to L1 (Σ|w| = max_gross)
    sum_abs = z.abs().sum(axis=1)
    w = z.div(sum_abs.replace(0.0, np.nan), axis=0) * max_gross

    # Preserve NaN on warm-up rows for returned weights
    w.loc[warmup] = np.nan

    # Use 0s only for math
    w_math = w.fillna(0.0)
    w_prev = w_math.shift(1).fillna(0.0)

    turnover = (w_math - w_prev).abs().sum(axis=1)
    cost = turnover * (cost_bps / 10000.0)

    gross_returns = (w_math * rets).sum(axis=1)
    net_returns = gross_returns - cost

    return {
        "weights": w,              # warm-up stays NaN here
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "cost": cost,
    }
