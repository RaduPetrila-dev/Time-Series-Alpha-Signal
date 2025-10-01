from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _l1_normalize(weights: pd.DataFrame, target_gross: float) -> pd.DataFrame:
    """
    Normalize row-wise so that sum(|w_i|) == target_gross for each date.
    If a row sums to 0 (or all-NaN), the result for that row stays NaN.
    """
    gross = weights.abs().sum(axis=1)
    # Replace zeros with NaN to avoid division-by-zero; preserve NaNs for warm-up rows
    gross = gross.replace(0.0, np.nan)
    return weights.div(gross, axis=0) * float(target_gross)


def backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: float = 10.0,
    max_gross: float = 1.0,
    lag_signal: int = 1,
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Vectorized backtest with:
      - No look-ahead (signals are lagged by `lag_signal`)
      - L1 exposure control (Σ|w| = max_gross per day)
      - Proportional transaction costs based on turnover

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price matrix (columns = tickers, index = dates).
    signal : pd.DataFrame
        Raw desired signal (any scale). Will be lagged and L1-normalized.
        NOTE: We do NOT cross-sectionally standardize by default so that
        single-asset tests behave intuitively.
    cost_bps : float
        Transaction cost in basis points per unit turnover (e.g. 10 = 0.001).
    max_gross : float
        Daily L1 exposure target (Σ|weights|).
    lag_signal : int
        Number of days to lag the signal to prevent look-ahead.

    Returns
    -------
    dict with keys:
        - "weights"        : pd.DataFrame of portfolio weights (first `lag_signal` rows are NaN)
        - "gross_returns"  : pd.Series of Σ(w * returns)
        - "net_returns"    : pd.Series of gross - cost
        - "turnover"       : pd.Series of Σ|Δw|
        - "cost"           : pd.Series of turnover * (cost_bps / 10_000)
    """
    if not isinstance(prices, pd.DataFrame) or not isinstance(signal, pd.DataFrame):
        raise TypeError("prices and signal must be pandas DataFrames")

    # 1) Align on dates/tickers
    prices, signal = prices.align(signal, join="inner")
    prices = prices.astype(float)

    # 2) Compute simple returns
    rets = prices.pct_change().fillna(0.0)

    # 3) Lag signal to enforce no-lookahead
    sig_lagged = signal.shift(lag_signal)

    # 4) L1-normalize to target gross exposure
    w = _l1_normalize(sig_lagged, max_gross=max_gross)

    # 5) Explicitly mark first `lag_signal` rows as NaN (so tests can detect lag)
    if lag_signal > 0:
        w.iloc[:lag_signal] = np.nan

    # 6) Use zero-filled copy for return/cost/turnover math
    w_math = w.fillna(0.0)
    w_prev = w_math.shift(1).fillna(0.0)

    # 7) Turnover & costs
    turnover = (w_math - w_prev).abs().sum(axis=1)
    cost = turnover * (cost_bps / 10_000.0)

    # 8) Portfolio returns
    gross_returns = (w_math * rets).sum(axis=1)
    net_returns = gross_returns - cost

    return {
        "weights": w,                # first `lag_signal` rows remain NaN
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "cost": cost,
    }

