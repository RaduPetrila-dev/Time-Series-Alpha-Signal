from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def daily_volatility(prices: pd.Series, span: int = 100) -> pd.Series:
    """Estimate daily volatility using an exponential moving standard deviation.

    This helper computes the rolling standard deviation of daily returns
    using an exponential weighting scheme.  A larger ``span`` produces
    a smoother estimate that reacts more slowly to changes in
    volatility.  The resulting series is aligned to the input
    ``prices`` index and forward‑filled for any initial missing values.

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
        Estimated daily volatility of returns, forward‑filled for
        leading NaNs.
    """
    returns = prices.pct_change()
    vol = returns.ewm(span=span, adjust=False).std()
    return vol.fillna(method="bfill")


def get_vertical_barriers(prices: pd.Series, horizon: int) -> pd.Series:
    """Locate the vertical barrier (time limit) for each event.

    For each timestamp in ``prices``, this function computes the end time
    at a fixed forward horizon.  If the horizon extends beyond the
    available data, the end time is set to the last index.  This helper
    is used internally by :func:`get_events` to define the maximum
    holding period for each event.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    horizon : int
        Number of observations between the event start and the vertical
        barrier.  For daily data a horizon of 20 means the event is
        evaluated over the next 20 trading days.

    Returns
    -------
    Series
        Series mapping each start time to the corresponding end time.
    """
    idx = prices.index
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    # For each timestamp find the index position horizon steps ahead
    pos = np.arange(len(idx)) + horizon
    pos[pos >= len(idx)] = len(idx) - 1
    return pd.Series(idx[pos], index=idx)


def get_events(
    prices: pd.Series,
    vol: pd.Series,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    horizon: int = 20,
) -> pd.DataFrame:
    """Construct the events table for the triple‑barrier method.

    An event begins at each observation in ``prices`` and ends when one
    of three conditions is met: the price hits a profit‑taking or stop‑loss
    barrier scaled by the local volatility, or a fixed time horizon is
    reached.  The barriers are determined by ``pt_sl``, which specify
    multipliers on the volatility estimate ``vol``.  Events with missing
    volatility are omitted.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    vol : Series
        Volatility estimate aligned to ``prices`` (e.g. from
        :func:`daily_volatility`).  Events with missing or non‑positive
        volatility are ignored.
    pt_sl : tuple of float, default (1.0, 1.0)
        Multipliers for the profit‑taking and stop‑loss barriers.
    horizon : int, default 20
        Maximum number of periods to hold the position.  The vertical
        barrier is located at ``t + horizon``.

    Returns
    -------
    DataFrame
        Events table with columns ``t1`` (vertical barrier time),
        ``pt`` (profit‑taking price) and ``sl`` (stop‑loss price).  The
        index represents the event start time ``t0``.
    """
    if len(prices) != len(vol):
        raise ValueError("prices and vol must be of the same length")
    vb = get_vertical_barriers(prices, horizon)
    df0 = pd.DataFrame(index=prices.index)
    df0["t1"] = vb
    # scale barriers by volatility; forward‑fill vol to handle leading NaNs
    vol_filled = vol.fillna(method="bfill").fillna(method="ffill")
    df0["pt"] = prices * (1 + pt_sl[0] * vol_filled)
    df0["sl"] = prices * (1 - pt_sl[1] * vol_filled)
    # drop rows where volatility is non‑positive (invalid)
    df0 = df0[vol_filled > 0]
    return df0


def apply_triple_barrier(
    prices: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Assign labels to events using the triple‑barrier method.

    For each event defined by ``events`` this function scans the price
    path between the start time ``t0`` and the vertical barrier ``t1``.
    If the price exceeds the profit‑taking level before hitting the
    stop‑loss level the label is +1.  If the stop‑loss is hit first the
    label is −1.  Otherwise the label is the sign of the return at the
    vertical barrier.  The resulting DataFrame has the same index as
    ``events`` and columns ``y`` (label) and ``t1`` (end time).

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    events : DataFrame
        Event definitions with columns ``t1``, ``pt`` and ``sl``.

    Returns
    -------
    DataFrame
        Events with an additional column ``y`` containing the labels.
    """
    out = events.copy()
    # allocate label array
    labels = np.zeros(len(out), dtype=int)
    # iterate over events
    for i, (t0, row) in enumerate(out.iterrows()):
        t1 = row["t1"]
        pt = row["pt"]
        sl = row["sl"]
        # ensure t0 < t1
        if t1 <= t0:
            labels[i] = 0
            continue
        # price window from t0 to t1 inclusive
        window = prices.loc[t0:t1]
        # skip if empty
        if window.empty:
            labels[i] = 0
            continue
        hit_pt = (window >= pt).any()
        hit_sl = (window <= sl).any()
        if hit_pt and not hit_sl:
            labels[i] = 1
        elif hit_sl and not hit_pt:
            labels[i] = -1
        else:
            # vertical barrier: assign sign of return
            labels[i] = int(np.sign(window.iloc[-1] - prices.loc[t0]))
    out = out.rename(columns={"pt": "pt", "sl": "sl"})
    out["y"] = labels
    return out[["t1", "y"]]


def triple_barrier_labels(
    prices: pd.Series,
    vol: pd.Series | None = None,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    horizon: int = 20,
) -> pd.DataFrame:
    """Apply the triple‑barrier method to generate event labels.

    A wrapper around :func:`get_events` and :func:`apply_triple_barrier`
    that handles volatility estimation and missing values.  If ``vol``
    is not provided the daily volatility is estimated via
    :func:`daily_volatility`.  The profit‑taking and stop‑loss barriers
    are scaled by ``pt_sl`` and the time limit is set by ``horizon``.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    vol : Series, optional
        Volatility estimate aligned to ``prices``.  If ``None`` the
        volatility is computed automatically.
    pt_sl : tuple of float, default (1.0, 1.0)
        Multipliers for the profit‑taking and stop‑loss barriers.
    horizon : int, default 20
        Maximum number of periods to hold the position.

    Returns
    -------
    DataFrame
        DataFrame with columns ``t1`` and ``y`` (label).  The index
        represents the event start time ``t0``.
    """
    # compute volatility if not provided
    if vol is None:
        vol = daily_volatility(prices)
    # construct events table
    events = get_events(prices, vol, pt_sl=pt_sl, horizon=horizon)
    # drop events whose horizon goes beyond available data
    events = events[events.index < events["t1"]]
    # apply triple barrier logic to assign labels
    labels_df = apply_triple_barrier(prices, events)
    return labels_df


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
