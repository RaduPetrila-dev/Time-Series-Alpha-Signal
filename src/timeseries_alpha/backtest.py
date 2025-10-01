from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _l1_normalize(rows: pd.DataFrame, target_gross: float) -> pd.DataFrame:
    """
    Row-wise L1 normalization so that sum(|w_i|) == target_gross for each date.
    If a row is all-NaN or sums to 0, it stays NaN to reflect 'no signal'.
    """
    gross = rows.abs().sum(axis=1)          # skipna=True by default
    gross = gross.replace(0.0, np.nan)      # avoid div-by-zero; keep NaN for empty rows
    return rows.div(gross, axis=0) * float(target_gross)


def backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: float = 10.0,
    max_gross: float = 1.0,
    lag_signal: int = 1,   # <-- default MUST be 1 for your test
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Vectorized backtest:
      - No look-ahead (signals lagged by `lag_signal`)
      - L1 exposure control (Σ|w| = max_gross per day)
      - Proportional costs on turnover (Σ|Δw| * bps/10_000)

    Returns dict with:
      weights (DataFrame), gross_returns (Series), net_returns (Series),
      turnover (Series), cost (Series)
    """
    if not isinstance(prices, pd.DataFrame) or not isinstance(signal, pd.DataFrame):
        raise TypeError("prices and signal must be pandas DataFrames")

    # Align and ensure float
    prices, signal = prices.align(signal, join="inner")
    prices = prices.astype(float)
    signal = signal.astype(float)

    # Simple returns
    rets = prices.pct_change().fillna(0.0)

    # --- NO LOOK-AHEAD ---
    sig_lagged = signal.shift(lag_signal)

    # Normalize to target gross via L1 norm, preserving NaNs
    w = _l1_normalize(sig_lagged, target_gross=max_gross)

    # Explicitly mark the first `lag_signal` rows as NaN so the test can detect the lag
    if lag_signal > 0 and len(w) >= lag_signal:
        w.iloc[:lag_signal] = np.nan

    # Use a zero-filled copy for math only
    w_math = w.fillna(0.0)
    w_prev = w_math.shift(1).fillna(0.0)

    turnover = (w_math - w_prev).abs().sum(axis=1)
    cost = turnover * (cost_bps / 10_000.0)

    gross_returns = (w_math * rets).sum(axis=1)
    net_returns = gross_returns - cost

    # Ensure float dtype (helps DataFrame.dropna semantics)
    w = w.astype(float)

    return {
        "weights": w,                # <-- NaNs preserved on warm-up rows
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "cost": cost,
    }

    
