
"""
signals.py — alpha signal functions
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def _zscore_cross_sectional(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Cross-sectional zscore by date (row-wise standardization)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = (df.sub(mean, axis=0)).div(std + eps, axis=0)
    return z.fillna(0.0)

def momentum_signal(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """
    Simple price momentum: (P / P_{t-LB}) - 1, then cross-sectional z-score each day.
    """
    if lookback <= 0:
        raise ValueError("lookback must be > 0")
    mom = prices / prices.shift(lookback) - 1.0
    # Standardize cross-sectionally per day
    sig = _zscore_cross_sectional(mom)
    return sig

def mean_reversion_zscore(returns: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Negative z-score of rolling returns — higher => more oversold (buy)."""
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    roll_mean = returns.rolling(lookback, min_periods=max(2, lookback//2)).mean()
    roll_std = returns.rolling(lookback, min_periods=max(2, lookback//2)).std(ddof=0)
    z = (returns - roll_mean) / (roll_std.replace(0.0, np.nan) + 1e-9)
    return (-z).fillna(0.0)

def combine_signals(
    signals: list[pd.DataFrame], 
    weights: list[float] | None = None
) -> pd.DataFrame:
    """Weighted average of multiple signals with safe alignment."""
    if not signals:
        raise ValueError("signals list is empty")
    base = signals[0].copy()
    for s in signals[1:]:
        base, s = base.align(s, join="outer")
        base = base.fillna(0.0)
        s = s.fillna(0.0)
        base = base.add(s, fill_value=0.0)
    if weights:
        if len(weights) != len(signals):
            raise ValueError("weights length must match signals length")
        # recompute using weights
        base = None
        total = None
        for s, w in zip(signals, weights):
            s = s.reindex(index=signals[0].index, columns=signals[0].columns).fillna(0.0)
            part = s * float(w)
            base = part if base is None else base.add(part, fill_value=0.0)
            total = float(w) if total is None else total + float(w)
        return base / (total if total else 1.0)
    # simple average
    return base / float(len(signals))
