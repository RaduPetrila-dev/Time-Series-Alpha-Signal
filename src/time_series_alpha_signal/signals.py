"""Cross-sectional signal functions for the backtesting pipeline.

Each signal function accepts a price DataFrame (datetime index, asset
columns) and returns a DataFrame of the same shape containing
cross-sectional scores.  All signals are **lagged by one period** to
prevent lookahead bias.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _validate_prices(prices: pd.DataFrame, min_rows: int = 2) -> None:
    if prices.empty or len(prices) < min_rows:
        raise ValueError(f"prices must have >= {min_rows} rows, got {len(prices)}.")
    if prices.shape[1] == 0:
        raise ValueError("prices must have at least one asset column.")


def _validate_lookback(lookback: int, name: str = "lookback") -> None:
    if lookback < 1:
        raise ValueError(f"{name} must be >= 1, got {lookback}.")


def momentum_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_lookback(lookback)
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    return mom.shift(1)


def mean_reversion_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_lookback(lookback)
    rets = prices.pct_change()
    rev = -rets.rolling(lookback).sum()
    return rev.shift(1)


def skip_month_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Cross-sectional momentum skipping the most recent month.

    Computes the return from *lookback* days ago to *skip* days ago,
    excluding the most recent *skip* days to avoid short-term reversal
    contamination (Jegadeesh, 1990).

    The classic 12-1 configuration uses lookback=252, skip=21.

    Parameters
    ----------
    prices : DataFrame
        Price data (datetime index, asset columns).
    lookback : int
        Total formation period in trading days (default 252).
    skip : int
        Recent days to exclude (default 21, ~1 month).

    Returns
    -------
    DataFrame
        Cross-sectional signal scores (lagged by 1 day).
    """
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(skip, "skip")
    if skip >= lookback:
        raise ValueError(f"skip ({skip}) must be < lookback ({lookback}).")
    formation_ret = prices.shift(skip) / prices.shift(lookback) - 1
    return formation_ret.shift(1)


def residual_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Cross-sectional residual momentum (idiosyncratic momentum).

    For each day, regresses each asset's trailing returns against the
    equal-weight market return to obtain residuals. Then computes
    momentum on the residual returns, skipping the most recent *skip*
    days to avoid short-term reversal.

    This captures stock-specific trends independent of broad market
    direction, reducing drawdowns during market-wide selloffs.

    Parameters
    ----------
    prices : DataFrame
        Price data (datetime index, asset columns).
    lookback : int
        Formation period in trading days (default 252).
    skip : int
        Recent days to exclude (default 21).

    Returns
    -------
    DataFrame
        Cross-sectional residual momentum scores (lagged by 1 day).

    References
    ----------
    Blitz, D., Huij, J. & Martens, M. (2011). Residual Momentum.
    *Journal of Empirical Finance*, 18(3), 506-521.
    """
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(skip, "skip")
    if skip >= lookback:
        raise ValueError(f"skip ({skip}) must be < lookback ({lookback}).")

    rets = prices.pct_change()
    market_ret = rets.mean(axis=1)

    residuals = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    rolling_var = market_ret.rolling(lookback, min_periods=lookback // 2).var()

    for col in rets.columns:
        rolling_cov = rets[col].rolling(lookback, min_periods=lookback // 2).cov(market_ret)
        beta = rolling_cov / rolling_var.clip(lower=1e-10)
        residuals[col] = rets[col] - beta * market_ret

    residual_cum = residuals.rolling(lookback, min_periods=lookback // 2).sum()
    residual_cum_skip = residuals.rolling(skip, min_periods=1).sum()
    signal = residual_cum - residual_cum_skip

    return signal.shift(1)


def arima_signal(
    prices: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 1),
) -> pd.DataFrame:
    _validate_prices(prices)
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is required for arima_signal. Install with: pip install statsmodels"
        ) from exc
    rets = prices.pct_change().dropna()
    signals = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
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


def volatility_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_lookback(lookback)
    rets = prices.pct_change()
    vol = rets.pow(2).rolling(lookback).mean()
    return (-vol).shift(1)


def volatility_scaled_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(vol_window, "vol_window")
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    vol = rets.rolling(vol_window).std()
    scaled = mom / (vol + _EPS)
    return scaled.shift(1)


def regime_switch_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(vol_window, "vol_window")
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    vol = rets.rolling(vol_window).std()
    avg_vol = vol.mean(axis=1)
    low_vol_mask = avg_vol < vol_threshold
    signal = mom.where(low_vol_mask, -mom)
    signal[avg_vol.isna()] = np.nan
    n_momentum = int(low_vol_mask.sum())
    n_meanrev = int((~low_vol_mask & ~avg_vol.isna()).sum())
    logger.debug("Regime switch: %d momentum days, %d mean-reversion days.", n_momentum, n_meanrev)
    return signal.shift(1)


def ewma_momentum_signal(prices: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_lookback(span, "span")
    rets = prices.pct_change()
    ewma = rets.ewm(span=span, adjust=False).mean()
    return ewma.shift(1)


def moving_average_crossover_signal(
    prices: pd.DataFrame,
    ma_short: int = 10,
    ma_long: int = 50,
) -> pd.DataFrame:
    _validate_prices(prices)
    _validate_lookback(ma_short, "ma_short")
    _validate_lookback(ma_long, "ma_long")
    if ma_long <= ma_short:
        raise ValueError(f"ma_long ({ma_long}) must be > ma_short ({ma_short}).")
    ma_s = prices.rolling(ma_short).mean()
    ma_l = prices.rolling(ma_long).mean()
    return (ma_s - ma_l).shift(1)
