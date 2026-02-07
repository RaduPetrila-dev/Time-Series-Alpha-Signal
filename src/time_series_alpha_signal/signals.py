"""Cross-sectional signal functions for the backtesting pipeline.

Each signal function accepts a price DataFrame (datetime index, asset
columns) and returns a DataFrame of the same shape containing
cross-sectional scores.  All signals are **lagged by one period** to
prevent lookahead bias.

Signals are registered in the backtest module's signal registry at
import time.  To add a custom signal, define a function with signature
``(prices: DataFrame, **kwargs) -> DataFrame`` and call
:func:`~time_series_alpha_signal.backtest.register_signal`.

Available signals
-----------------
* :func:`momentum_signal` -- trailing return sum.
* :func:`mean_reversion_signal` -- negative trailing return sum.
* :func:`arima_signal` -- ARIMA one-step forecasts (slow).
* :func:`volatility_signal` -- negative rolling variance (low-vol tilt).
* :func:`volatility_scaled_momentum_signal` -- momentum / rolling vol.
* :func:`regime_switch_signal` -- momentum or mean-reversion depending
  on universe volatility.
* :func:`ewma_momentum_signal` -- exponentially weighted momentum.
* :func:`moving_average_crossover_signal` -- short MA minus long MA.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Small constant to avoid division by zero in volatility scaling.
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_prices(prices: pd.DataFrame, min_rows: int = 2) -> None:
    """Check that *prices* is a non-empty DataFrame with enough rows.

    Raises
    ------
    ValueError
        If *prices* has fewer than *min_rows* rows or no columns.
    """
    if prices.empty or len(prices) < min_rows:
        raise ValueError(
            f"prices must have >= {min_rows} rows, got {len(prices)}."
        )
    if prices.shape[1] == 0:
        raise ValueError("prices must have at least one asset column.")


def _validate_lookback(lookback: int, name: str = "lookback") -> None:
    """Ensure *lookback* is a positive integer."""
    if lookback < 1:
        raise ValueError(f"{name} must be >= 1, got {lookback}.")


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Trailing return sum over *lookback* periods.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Number of periods to sum returns over.

    Returns
    -------
    DataFrame
        Lagged momentum scores.
    """
    _validate_prices(prices)
    _validate_lookback(lookback)

    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    return mom.shift(1)


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------


def mean_reversion_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Negative trailing return sum (contrarian signal).

    Assets that outperformed over *lookback* receive negative scores;
    underperformers receive positive scores.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Lookback period.

    Returns
    -------
    DataFrame
        Lagged mean-reversion scores.
    """
    _validate_prices(prices)
    _validate_lookback(lookback)

    rets = prices.pct_change()
    rev = -rets.rolling(lookback).sum()
    return rev.shift(1)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------


def arima_signal(
    prices: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 0, 1),
) -> pd.DataFrame:
    """One-step-ahead ARIMA forecasts of returns.

    Fits an ARIMA model per asset on the full return history and uses
    the in-sample predictions as scores.  Falls back to a rolling mean
    if the model fails to converge.

    .. warning::
       This signal fits one model per asset and is slow on large
       universes.  Use with care.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    order : tuple of int, default (1, 0, 1)
        ARIMA ``(p, d, q)`` order.

    Returns
    -------
    DataFrame
        Lagged ARIMA forecast scores.
    """
    _validate_prices(prices)

    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is required for arima_signal. "
            "Install with: pip install statsmodels"
        ) from exc

    rets = prices.pct_change().dropna()
    signals = pd.DataFrame(
        index=rets.index, columns=rets.columns, dtype=float
    )

    n_fallback = 0
    for col in rets.columns:
        series = rets[col]
        try:
            model = ARIMA(series, order=order)
            fit = model.fit()
            signals[col] = fit.predict()
        except Exception:
            window = max(order[0], 1)
            signals[col] = series.rolling(window).mean()
            n_fallback += 1

    if n_fallback > 0:
        logger.warning(
            "ARIMA fit failed for %d/%d assets; used rolling mean fallback.",
            n_fallback,
            len(rets.columns),
        )

    return signals.shift(1)


# ---------------------------------------------------------------------------
# Volatility (low-vol tilt)
# ---------------------------------------------------------------------------


def volatility_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Negative rolling variance (low-volatility tilt).

    Low-volatility assets receive positive scores; high-volatility
    assets receive negative scores.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Window for rolling variance.

    Returns
    -------
    DataFrame
        Lagged volatility scores.
    """
    _validate_prices(prices)
    _validate_lookback(lookback)

    rets = prices.pct_change()
    vol = rets.pow(2).rolling(lookback).mean()
    return (-vol).shift(1)


# ---------------------------------------------------------------------------
# Volatility-scaled momentum
# ---------------------------------------------------------------------------


def volatility_scaled_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    r"""Momentum divided by rolling volatility.

    Down-weights high-volatility assets by scaling the trailing return
    sum by the inverse of rolling standard deviation:

    .. math::
        s_{i,t} = \frac{\sum_{k=1}^{L} r_{i,t-k}}
                       {\hat\sigma_{i,t} + \epsilon}

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Momentum lookback.
    vol_window : int, default 20
        Rolling std window.

    Returns
    -------
    DataFrame
        Lagged vol-scaled momentum scores.
    """
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(vol_window, "vol_window")

    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    vol = rets.rolling(vol_window).std()
    scaled = mom / (vol + _EPS)
    return scaled.shift(1)


# ---------------------------------------------------------------------------
# Regime switch
# ---------------------------------------------------------------------------


def regime_switch_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
) -> pd.DataFrame:
    """Momentum or mean-reversion depending on universe volatility.

    When the cross-sectional average of rolling volatility is below
    *vol_threshold*, the momentum signal is used.  Above the threshold,
    the mean-reversion signal is applied.

    The regime detection and signal selection are fully **vectorised**
    (no row-level Python loop).

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Trailing return lookback.
    vol_window : int, default 20
        Rolling std window for regime detection.
    vol_threshold : float, default 0.02
        Average volatility cutoff.

    Returns
    -------
    DataFrame
        Lagged regime-switching scores.
    """
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(vol_window, "vol_window")

    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()

    # Rolling volatility per asset, then cross-sectional mean
    vol = rets.rolling(vol_window).std()
    avg_vol = vol.mean(axis=1)

    # Vectorised regime mask: True = low-vol (momentum), False = high-vol (mean-rev)
    low_vol_mask = avg_vol < vol_threshold

    # Broadcast the mask across columns
    # momentum where low-vol, negative momentum (mean-rev) where high-vol
    signal = mom.where(low_vol_mask, -mom)

    # NaN out rows where avg_vol is NaN
    signal[avg_vol.isna()] = np.nan

    n_momentum = int(low_vol_mask.sum())
    n_meanrev = int((~low_vol_mask & ~avg_vol.isna()).sum())
    logger.debug(
        "Regime switch: %d momentum days, %d mean-reversion days.",
        n_momentum,
        n_meanrev,
    )

    return signal.shift(1)


# ---------------------------------------------------------------------------
# EWMA momentum
# ---------------------------------------------------------------------------


def ewma_momentum_signal(
    prices: pd.DataFrame,
    span: int = 20,
) -> pd.DataFrame:
    """Exponentially weighted moving average of returns.

    Places more weight on recent returns than the simple sum used by
    :func:`momentum_signal`.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    span : int, default 20
        EWMA span.  Larger = slower decay.

    Returns
    -------
    DataFrame
        Lagged EWMA momentum scores.
    """
    _validate_prices(prices)
    _validate_lookback(span, "span")

    rets = prices.pct_change()
    ewma = rets.ewm(span=span, adjust=False).mean()
    return ewma.shift(1)


# ---------------------------------------------------------------------------
# Moving average crossover
# ---------------------------------------------------------------------------


def moving_average_crossover_signal(
    prices: pd.DataFrame,
    ma_short: int = 10,
    ma_long: int = 50,
) -> pd.DataFrame:
    """Short MA minus long MA as a trend indicator.

    A positive value means the short-term trend is above the
    long-term trend (bullish).

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    ma_short : int, default 10
        Short moving average window.
    ma_long : int, default 50
        Long moving average window.  Must be > *ma_short*.

    Returns
    -------
    DataFrame
        Lagged crossover scores.

    Raises
    ------
    ValueError
        If *ma_long* <= *ma_short*.
    """
    _validate_prices(prices)
    _validate_lookback(ma_short, "ma_short")
    _validate_lookback(ma_long, "ma_long")

    if ma_long <= ma_short:
        raise ValueError(
            f"ma_long ({ma_long}) must be > ma_short ({ma_short})."
        )

    ma_s = prices.rolling(ma_short).mean()
    ma_l = prices.rolling(ma_long).mean()
    return (ma_s - ma_l).shift(1)
