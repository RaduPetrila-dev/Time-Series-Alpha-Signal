"""
metrics.py — portfolio statistics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, annualization: int = 252) -> float:
    mu = returns.mean() * annualization
    sd = returns.std(ddof=0) * np.sqrt(annualization)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(mu / sd)

def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return float(dd.min())

def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() * start_value

def avg_turnover(turnover: pd.Series) -> float:
    return float(turnover.mean())
