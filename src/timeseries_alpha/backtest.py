
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd



def _align_frames(*dfs: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """
    Return the common (index, columns) across provided DataFrames.
    """
    idx = dfs[0].index
    cols = dfs[0].columns
    for df in dfs[1:]:
        idx = idx.intersection(df.index)
        cols = cols.intersection(df.columns)
    return idx.sort_values(), cols.sort_values()


def _safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinities with NaN; keep dtype stable.
    """
    return df.replace([np.inf, -np.inf], np.nan)


def _row_l1_norm(df: pd.DataFrame) -> pd.Series:
    return df.abs().sum(axis=1)


def prepare_weights_from_signal(
    signal: pd.DataFrame,
    tradable: Optional[pd.DataFrame] = None,
    max_gross: float = 1.0,
    clip_at: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convert any real-valued cross-sectional alpha 'signal' into target weights w,
    enforcing Σ|w| = max_gross (L1 budget) per date. Handles NaN alignment.

    Parameters
    ----------
    signal : DataFrame (t x n)
        Larger positive => larger long weight; negative => short.
    tradable : DataFrame (t x n), optional
        Boolean mask; False => force weight to 0. If None, uses ~signal.isna().
    max_gross : float
        L1 exposure per row after scaling.
    clip_at : float, optional
        If given, clip raw signal to [-clip_at, clip_at] before scaling.

    Returns
    -------
    DataFrame of weights with row-wise L1 max_gross and NaNs converted to 0.
    """
    s = _safe(signal.copy())
    if clip_at is not None:
        s = s.clip(lower=-clip_at, upper=clip_at)

    if tradable is None:
        tradable = ~s.isna()

    s = s.where(tradable, 0.0).fillna(0.0)

    # If an entire row is zero, keep it zero; otherwise scale to L1 = max_gross
    l1 = _row_l1_norm(s)
    nonzero = l1 > 0
    w = s.copy()
    w.loc[nonzero] = w.loc[nonzero].div(l1[nonzero], axis=0) * max_gross
    return w.astype(float)


def _turnover(
    w_target: pd.DataFrame,
    w_prev: pd.DataFrame,
    use_drift: bool,
    realized_ret: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    One-way turnover per date.

    If use_drift is True, compare target weights to *post-return drifted* weights
    (more realistic); else compare to prior target directly.

    Returns
    -------
    Series of one-way turnover (Σ |Δw|), not divided by 2.
    """
    if use_drift:
        if realized_ret is None:
            raise ValueError("realized_ret required when use_drift=True")
        # Drift prior weights by realized returns, then renormalize to keep L1 gross the 
        # same as previous.
        # Avoid division by zero by handling empty rows.
        w_pretrade = (w_prev * (1.0 + realized_ret)).fillna(0.0)
        tgt_l1 = _row_l1_norm(w_prev)
        # scale back to original L1 (so drifted notional totals match prior gross)
        scale = (
            tgt_l1.divide(_row_l1_norm(w_pretrade))
            .replace([np.inf, -np.inf], np.nan).
            fillna(0.0)
        )
        w_pretrade = w_pretrade.mul(scale, axis=0).fillna(0.0)
    else:
        w_pretrade = w_prev.fillna(0.0)

    return (w_target.fillna(0.0) - w_pretrade).abs().sum(axis=1)


def backtest(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    max_gross: float = 1.0,
    clip_at: Optional[float] = None,
    cost_per_dollar: float = 0.0005,  # 5 bps one-way cost per unit turnover
    use_drift_for_turnover: bool = True,
    two_way_cost: bool = True,
) -> Dict[str, pd.DataFrame | pd.Series]:
    """
    Vectorized cross-sectional backtest.

    Conventions
    -----------
    - 'returns' are simple returns for each asset on date t.
    - We trade at the end of t-1 using signal_{t-1}, setting weights for t.
      No-lookahead: weights_t = f(signal_{t-1}); PnL_t = (weights_t · returns_t) - costs_t.
    - Σ|weights_t| = max_gross when there is at least one tradable name.

    Parameters
    ----------
    returns : DataFrame (t x n)
    signal  : DataFrame (t x n)
    max_gross : float
    clip_at : float, optional
    cost_per_dollar : float
        Cost per unit absolute turnover. If two_way_cost=True, total daily cost is:
        cost_per_dollar * turnover_one_way.
    use_drift_for_turnover : bool
        If True, turnover compares target weights to drifted prior weights.
    two_way_cost : bool
        True leaves turnover as Σ|Δw| (one-way dollars traded). If you want to model
        cost per *side* separately, halve or adjust externally.

    Returns
    -------
    dict containing:
      - weights
      - gross_pnl (before costs)
      - cost
      - net_pnl
      - equity (cumprod on 1+net)
      - turnover
    """
    # Align and clean
    idx, cols = _align_frames(_safe(returns), _safe(signal))
    r = _safe(returns.loc[idx, cols]).astype(float)
    s = _safe(signal.loc[idx, cols]).astype(float)

    # No-lookahead: use previous day's signal to set today's weights
    s_lag = s.shift(1)
    tradable = ~r.isna()
    w = prepare_weights_from_signal(s_lag.where(tradable), max_gross=max_gross, clip_at=clip_at)

    # Gross daily PnL from weights applied on same-date returns
    gross = (w * r.fillna(0.0)).sum(axis=1)

    # Turnover & costs
    w_prev = w.shift(1).fillna(0.0)
    t_over = _turnover(
        w_target=w,
        w_prev=w_prev,
        use_drift=use_drift_for_turnover,
        realized_ret=r if use_drift_for_turnover else None,
    )
    # Cost in return-space (cost_per_dollar * turnover). If you prefer two-way bps 
    # per round-trip, keep as-is.
    cost = cost_per_dollar * t_over
    net = gross - cost

    equity = (1.0 + net.fillna(0.0)).cumprod()

    out: Dict[str, pd.DataFrame | pd.Series] = {
        "weights": w,
        "gross_pnl": gross.rename("gross"),
        "cost": cost.rename("cost"),
        "net_pnl": net.rename("net"),
        "equity": equity.rename("equity"),
        "turnover": t_over.rename("turnover"),
    }
    return out

    
