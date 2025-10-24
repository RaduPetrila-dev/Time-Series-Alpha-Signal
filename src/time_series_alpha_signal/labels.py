from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def daily_volatility(prices: pd.Series, span: int = 100) -> pd.Series:
    """Estimate daily volatility using an exponential moving standard deviation.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    span : int, default 100
        Span parameter for the exponential moving standard deviation.  A
        larger span results in a smoother volatility estimate.

    Returns
    -------
    Series
        Estimated daily volatility of returns.
    """
    returns = prices.pct_change()
    vol = returns.ewm(span=span, adjust=False).std()
    return vol


def triple_barrier_labels(
    prices: pd.Series,
    vol: pd.Series,
    pt_sl: Tuple[float, float] = (3.0, 3.0),
    horizon: int = 20,
) -> pd.DataFrame:
    """Apply the triple‑barrier method to generate event labels.

    For each time ``t`` a profit‑taking and stop‑loss barrier are set at
    ``prices[t] * (1 ± pt_sl * vol[t])``.  The event is evaluated until
    either barrier is hit or a vertical barrier ``t + horizon`` is
    reached.  If the upper barrier is hit first the label is +1; if
    the lower barrier is hit first the label is −1; otherwise it is 0.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    vol : Series
        Volatility estimate aligned to ``prices`` (e.g. from
        :func:`daily_volatility`).
    pt_sl : tuple of float, default (3.0, 3.0)
        Multipliers for the profit‑taking and stop‑loss barriers.  The
        first value scales the upper barrier and the second the lower.
    horizon : int, default 20
        Maximum number of periods to hold the position.  The vertical
        barrier is located at ``t + horizon``.

    Returns
    -------
    DataFrame
        A dataframe with the columns ``label`` and ``t1`` (the end time
        of the event).  Rows correspond to the start times of each
        event.  Events whose horizon would extend beyond the length of
        the series are dropped.
    """
    if len(prices) != len(vol):
        raise ValueError("prices and vol must be of the same length")
    n = len(prices)
    labels: Dict[pd.Timestamp, Dict[str, object]] = {}
    # iterate through each possible event start except the last horizon points
    for i in range(n - horizon):
        t0 = prices.index[i]
        s0 = prices.iloc[i]
        v0 = vol.iloc[i]
        # skip if volatility is missing or zero
        if pd.isna(v0) or v0 <= 0:
            continue
        pt = s0 * (1 + pt_sl[0] * v0)
        sl = s0 * (1 - pt_sl[1] * v0)
        # default label is neutral
        label = 0
        t1 = prices.index[i + horizon]
        # scan forward until horizon
        window = prices.iloc[i + 1 : i + horizon + 1]
        for ts, price in window.items():
            if price >= pt:
                label = 1
                t1 = ts
                break
            if price <= sl:
                label = -1
                t1 = ts
                break
        labels[t0] = {"label": label, "t1": t1}
    df = pd.DataFrame.from_dict(labels, orient="index")
    df.index.name = "t0"
    return df


def meta_label(
    base_labels: pd.Series,
    realized_returns: pd.Series,
) -> pd.Series:
    """Compute meta‑labels for bet sizing.

    Given a series of base classification labels (+1, −1, 0) and the
    realised returns associated with each event, the meta‑label is 1 if
    the base label correctly predicts the direction of the return and 0
    otherwise.  Neutral events (label 0) are assigned 0.

    Parameters
    ----------
    base_labels : Series
        Base classification labels indexed by event start time ``t0``.
    realized_returns : Series
        Realised returns for each event, aligned to ``base_labels``.

    Returns
    -------
    Series
        Binary meta‑labels indicating whether the base prediction was
        directionally correct.
    """
    if not base_labels.index.equals(realized_returns.index):
        raise ValueError("base_labels and realized_returns must share the same index")
    meta = []
    for lbl, ret in zip(base_labels, realized_returns):
        if lbl == 0:
            meta.append(0)
        else:
            meta.append(int(np.sign(ret) == lbl))
    return pd.Series(meta, index=base_labels.index, dtype=int)
