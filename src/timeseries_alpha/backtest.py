"""
backtest.py — vectorized backtester with transaction costs and leverage cap
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def _normalize_to_gross(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    gross = weights.abs().sum(axis=1)
    scale = (max_gross / np.maximum(gross.to_numpy(), 1e-12))
    scale = np.minimum(scale, 1.0)
    return weights.mul(scale, axis=0)

def backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: float = 10.0,
    max_gross: float = 1.0,
    lag_signal: int = 1,
) -> dict:
    """
    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix.
    signal : pd.DataFrame
        Desired raw signal (un-normalized). Will be normalized cross-sectionally
        and scaled to 'max_gross'. Automatically lagged by `lag_signal` days to
        prevent lookahead.
    cost_bps : float
        Proportional transaction cost per unit turnover (e.g., 10 bps = 0.001).
    max_gross : float
        Cap on |long|+|short| exposure per day.
    lag_signal : int
        How many days to lag the signal before forming weights.
    """
    prices, signal = prices.align(signal, join="inner")
    rets = prices.pct_change().fillna(0.0)

    # standardize cross-sectionally (per day), then target gross exposure
    cs_std = signal.sub(signal.mean(axis=1), axis=0)
    denom = signal.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = cs_std.div(denom + 1e-9, axis=0).fillna(0.0)

    # lag to avoid lookahead
    z = z.shift(lag_signal)

    # normalize to max_gross using L1 norm (market-neutral style)
    sum_abs = z.abs().sum(axis=1).replace(0.0, np.nan)
    w = z.div(sum_abs, axis=0).fillna(0.0) * max_gross

    # apply turnover-based costs
    w_prev = w.shift(1).fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    cost = turnover * (cost_bps / 10000.0)

    port_ret_gross = (w * rets).sum(axis=1)
    port_ret_net = port_ret_gross - cost

    return {
        "weights": w,
        "gross_returns": port_ret_gross,
        "net_returns": port_ret_net,
        "turnover": turnover,
        "cost": cost,
    }
    
