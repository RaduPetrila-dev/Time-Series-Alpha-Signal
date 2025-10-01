
from __future__ import annotations

from typing import Dict

import pandas as pd
import numpy as np


def perf_summary(net_returns: pd.Series, freq: str = "D") -> Dict[str, float]:
    """
    Compute simple performance stats on a net return series.
    """
    r = pd.Series(net_returns).dropna()
    if r.empty:
        return dict(n=0, ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_dd=0.0)

    if freq.upper().startswith("D"):
        ann = 252
    elif freq.upper().startswith("W"):
        ann = 52
    elif freq.upper().startswith("M"):
        ann = 12
    else:
        ann = 252

    mean = r.mean()
    vol = r.std(ddof=1)
    ann_return = (1 + r).prod() ** (ann / r.shape[0]) - 1 if r.shape[0] > 0 else 0.0
    ann_vol = vol * np.sqrt(ann)
    sharpe = (mean * ann) / (ann_vol + 1e-12)

    # Max drawdown from equity curve
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()

    return dict(
        n=int(r.shape[0]),
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_dd=float(dd),
    )
