"""Triple-barrier labelling and meta-labels for event-driven strategies.

This module implements the **triple-barrier method** described in:

    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Wiley. Chapter 3.

The triple-barrier method assigns a label to each event based on which
of three conditions is met first:

1. **Profit-taking barrier** -- price rises above a threshold scaled by
   local volatility.  Label = +1.
2. **Stop-loss barrier** -- price falls below a threshold scaled by
   local volatility.  Label = -1.
3. **Vertical barrier** -- a fixed time horizon expires.  Label = sign
   of the return at expiry.

When *both* the profit-taking and stop-loss barriers are breached in the
same window the barrier that was hit **first** (earliest timestamp)
takes priority.  This is a correction over naive implementations that
check ``.any()`` without regard to ordering.

Meta-labels (:func:`meta_label`) indicate whether a base classifier's
directional prediction was correct, enabling a secondary model to learn
*bet sizing*.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Volatility estimation
# ---------------------------------------------------------------------------


def daily_volatility(
    prices: pd.Series,
    span: int = 100,
) -> pd.Series:
    """Estimate daily volatility via exponentially-weighted std of returns.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    span : int, default 100
        EWM span.  Larger values produce smoother estimates.

    Returns
    -------
    Series
        Daily volatility estimate, back-filled for leading NaNs.
    """
    returns = prices.pct_change()
    vol = returns.ewm(span=span, adjust=False).std()
    return vol.bfill()


# ---------------------------------------------------------------------------
# Barrier construction
# ---------------------------------------------------------------------------


def get_vertical_barriers(
    prices: pd.Series,
    horizon: int,
) -> pd.Series:
    """Map each timestamp to its vertical barrier (time limit).

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    horizon : int
        Forward look (number of index steps).

    Returns
    -------
    Series
        Maps each start time to the end time.

    Raises
    ------
    ValueError
        If *horizon* < 1.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    idx = prices.index
    pos = np.arange(len(idx)) + horizon
    pos = np.clip(pos, 0, len(idx) - 1)
    return pd.Series(idx[pos], index=idx)


def get_events(
    prices: pd.Series,
    vol: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    horizon: int = 20,
) -> pd.DataFrame:
    """Construct the events table for the triple-barrier method.

    Each event starts at a price observation and defines profit-taking
    and stop-loss levels scaled by local volatility, plus a vertical
    barrier at ``t + horizon``.

    Parameters
    ----------
    prices : Series
        Price series.
    vol : Series
        Volatility estimate aligned to *prices*.
    pt_sl : tuple of float, default (1.0, 1.0)
        ``(profit_taking_mult, stop_loss_mult)`` applied to *vol*.
    horizon : int, default 20
        Vertical barrier in index steps.

    Returns
    -------
    DataFrame
        Columns: ``t1`` (vertical barrier), ``pt`` (profit-taking
        price), ``sl`` (stop-loss price).  Index = event start ``t0``.
        Rows with non-positive volatility are dropped.
    """
    if len(prices) != len(vol):
        raise ValueError(f"prices ({len(prices)}) and vol ({len(vol)}) length mismatch.")

    vb = get_vertical_barriers(prices, horizon)
    vol_filled = vol.bfill().ffill()

    df = pd.DataFrame(
        {
            "t1": vb,
            "pt": prices * (1 + pt_sl[0] * vol_filled),
            "sl": prices * (1 - pt_sl[1] * vol_filled),
        },
        index=prices.index,
    )

    # Drop invalid rows
    valid = vol_filled > 0
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logger.debug("Dropped %d events with non-positive volatility.", n_dropped)
    return df.loc[valid]


# ---------------------------------------------------------------------------
# Triple-barrier labelling
# ---------------------------------------------------------------------------


def _label_single_event(
    prices: pd.Series,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    pt: float,
    sl: float,
) -> int:
    """Assign a label to a single event based on barrier hits.

    When both barriers are hit in the same window, the one hit
    **first** (earliest index position) wins.

    Returns
    -------
    int
        +1 (profit-taking), -1 (stop-loss), or sign of terminal return.
    """
    if t1 <= t0:
        return 0

    window = prices.loc[t0:t1]
    if window.empty:
        return 0

    # Find the first time each barrier is breached (NaT if never)
    pt_hits = window.index[window >= pt]
    sl_hits = window.index[window <= sl]

    first_pt = pt_hits[0] if len(pt_hits) > 0 else pd.NaT
    first_sl = sl_hits[0] if len(sl_hits) > 0 else pd.NaT

    # Determine which barrier was hit first
    pt_hit = not pd.isna(first_pt)
    sl_hit = not pd.isna(first_sl)

    if pt_hit and sl_hit:
        # Both hit: earliest wins
        if first_pt <= first_sl:
            return 1
        return -1
    if pt_hit:
        return 1
    if sl_hit:
        return -1

    # Vertical barrier: sign of return
    terminal_return = window.iloc[-1] - prices.loc[t0]
    return int(np.sign(terminal_return))


def apply_triple_barrier(
    prices: pd.Series,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Assign labels to events using the triple-barrier method.

    For each event the price path is scanned from ``t0`` to ``t1``.
    The label is determined by which barrier is hit first:

    * +1 if the profit-taking barrier is hit first.
    * -1 if the stop-loss barrier is hit first.
    * sign(return) at the vertical barrier if neither is hit.

    When **both** barriers are breached within the window, the one
    with the **earlier** timestamp takes priority.

    Parameters
    ----------
    prices : Series
        Full price series.
    events : DataFrame
        Must contain ``t1``, ``pt``, ``sl`` columns.  Index = ``t0``.

    Returns
    -------
    DataFrame
        Columns ``t1`` and ``y`` (label), indexed by ``t0``.
    """
    labels = np.empty(len(events), dtype=int)

    for i, (t0, row) in enumerate(events.iterrows()):
        labels[i] = _label_single_event(prices, t0, row["t1"], row["pt"], row["sl"])

    result = pd.DataFrame(
        {"t1": events["t1"].values, "y": labels},
        index=events.index,
    )

    # Log label distribution
    counts = pd.Series(labels).value_counts().to_dict()
    logger.debug(
        "Triple-barrier labels: %d events, distribution: %s",
        len(labels),
        counts,
    )

    return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def triple_barrier_labels(
    prices: pd.Series,
    vol: pd.Series | None = None,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    horizon: int = 20,
    vol_span: int = 100,
) -> pd.DataFrame:
    """End-to-end triple-barrier labelling.

    Wraps :func:`daily_volatility`, :func:`get_events`, and
    :func:`apply_triple_barrier` into a single call.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    vol : Series, optional
        Pre-computed volatility.  If ``None``, estimated via
        :func:`daily_volatility` with *vol_span*.
    pt_sl : tuple of float, default (1.0, 1.0)
        ``(profit_taking_mult, stop_loss_mult)``.
    horizon : int, default 20
        Vertical barrier in index steps.
    vol_span : int, default 100
        EWM span for volatility estimation (used when *vol* is None).

    Returns
    -------
    DataFrame
        Columns ``t1`` (barrier time) and ``y`` (label), indexed by
        event start ``t0``.
    """
    if vol is None:
        vol = daily_volatility(prices, span=vol_span)

    events = get_events(prices, vol, pt_sl=pt_sl, horizon=horizon)

    # Drop events whose start is at or beyond the vertical barrier
    valid = events.index < events["t1"]
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logger.debug("Dropped %d events at/beyond the vertical barrier.", n_dropped)
    events = events.loc[valid]

    return apply_triple_barrier(prices, events)


# ---------------------------------------------------------------------------
# Meta-labels (vectorised)
# ---------------------------------------------------------------------------


def meta_label(
    base_labels: pd.Series,
    realized_returns: pd.Series,
) -> pd.Series:
    """Compute binary meta-labels for bet sizing.

    A meta-label is 1 if the base classifier's directional prediction
    matches the sign of the realised return, and 0 otherwise.  Neutral
    predictions (label = 0) always map to 0.

    Parameters
    ----------
    base_labels : Series
        Base classifier labels (+1, -1, or 0).
    realized_returns : Series
        Realised returns aligned to *base_labels*.

    Returns
    -------
    Series
        Binary meta-labels (0 or 1).

    Raises
    ------
    ValueError
        If the indices do not match.
    """
    if not base_labels.index.equals(realized_returns.index):
        raise ValueError("base_labels and realized_returns must share the same index.")

    lbl = base_labels.values
    ret_sign = np.sign(realized_returns.values)

    # Vectorised: correct when label != 0 and sign matches
    correct = (lbl != 0) & (ret_sign == lbl)
    meta = correct.astype(int)

    return pd.Series(meta, index=base_labels.index, dtype=int, name="meta_label")
