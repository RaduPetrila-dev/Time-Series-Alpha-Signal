from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # type: ignore

# A small constant to avoid division by zero in volatility scaling
_EPS = 1e-8


def momentum_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute a simple momentum signal based on trailing returns.

    The signal is the sum of past percentage changes over the lookback window.
    A lag of one period is applied to avoid look‑ahead bias.

    Parameters
    ----------
    prices : DataFrame
        Price series for each asset with datetime index and asset columns.
    lookback : int, default 20
        Number of periods over which to compute momentum.

    Returns
    -------
    DataFrame
        Lagged momentum scores aligned to weights computation.
    """
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    return mom.shift(1)


def mean_reversion_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute a simple mean‑reversion signal.

    This is the negative of the trailing return sum.  Assets that have
    outperformed over the lookback window receive negative scores and vice
    versa.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Lookback period for the trailing return sum.

    Returns
    -------
    DataFrame
        Lagged mean‑reversion scores.
    """
    rets = prices.pct_change()
    rev = -rets.rolling(lookback).sum()
    return rev.shift(1)


def arima_signal(
    prices: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 0, 1),
) -> pd.DataFrame:
    """Compute a signal based on ARIMA forecasts of returns.

    For each asset, an ARIMA model is fit to the full return series using
    the provided order.  The in‑sample one‑step ahead forecasts are used as
    signals, lagged by one period to prevent look‑ahead bias.  In cases
    where the model fails to fit (for example due to insufficient data), a
    rolling mean of the returns is used as a fallback.

    Note
    ----
    Fitting ARIMA models per asset can be computationally expensive on
    large universes.  Use with care.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    order : tuple of int, default (1, 0, 1)
        ARIMA (p,d,q) order.

    Returns
    -------
    DataFrame
        Lagged ARIMA forecast signals.
    """
    rets = prices.pct_change().dropna()
    signals = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for col in rets:
        series = rets[col]
        try:
            model = ARIMA(series, order=order)
            fit = model.fit()
            pred = fit.predict()
            signals[col] = pred
        except Exception:
            # fallback: simple rolling mean of returns
            window = max(order[0], 1)
            signals[col] = series.rolling(window).mean()
    return signals.shift(1)


def volatility_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute a volatility‑based signal (proxy for GARCH).

    This signal ranks assets by the negative of their rolling variance
    (squared returns) over a lookback window.  Low‑volatility names receive
    positive scores and high‑volatility names negative scores.  The result is
    lagged by one period to avoid look‑ahead.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Window length for computing rolling variance.

    Returns
    -------
    DataFrame
        Lagged volatility signals.
    """
    rets = prices.pct_change()
    vol = rets.pow(2).rolling(lookback).mean()
    return (-vol).shift(1)


def volatility_scaled_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Compute a momentum signal scaled by inverse volatility.

    This function divides the simple momentum signal by a rolling standard
    deviation (volatility) to down‑weight assets with high volatility.  A
    lag of one period is applied to the resulting signal to avoid look‑ahead
    bias.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Lookback period for the momentum component (sum of returns).
    vol_window : int, default 20
        Window length for computing rolling standard deviation.

    Returns
    -------
    DataFrame
        Lagged volatility‑scaled momentum scores.
    """
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    vol = rets.rolling(vol_window).std()
    # scale momentum by volatility plus epsilon to avoid division by zero
    scaled = mom.div(vol + _EPS)
    return scaled.shift(1)


def regime_switch_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
) -> pd.DataFrame:
    """Compute a regime‑switching signal between momentum and mean‑reversion.

    This signal monitors the realised volatility of the universe and chooses
    between momentum and mean‑reversion signals.  When the average rolling
    standard deviation across all assets is below ``vol_threshold`` the
    momentum signal is used; otherwise the negative of the momentum signal
    (mean‑reversion) is applied.  The output is lagged by one period.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Lookback period for computing trailing returns used in momentum and
        mean‑reversion components.
    vol_window : int, default 20
        Window length for computing rolling standard deviation used for regime
        detection.
    vol_threshold : float, default 0.02
        Threshold for average realised volatility (in absolute units).  If the
        average volatility at time ``t`` is below this value, momentum is
        applied; if above, mean‑reversion is applied.

    Returns
    -------
    DataFrame
        Lagged regime‑switching scores.
    """
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    # realized volatility (standard deviation)
    vol = rets.rolling(vol_window).std()
    avg_vol = vol.mean(axis=1)
    # allocate DataFrame for signals
    signal = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ts in prices.index:
        if pd.isna(avg_vol.loc[ts]):
            signal.loc[ts] = np.nan
        elif avg_vol.loc[ts] < vol_threshold:
            signal.loc[ts] = mom.loc[ts]
        else:
            signal.loc[ts] = -mom.loc[ts]
    return signal.shift(1)


def ewma_momentum_signal(prices: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """Compute an exponentially weighted momentum signal.

    This signal uses an exponential moving average (EWMA) of returns to
    capture momentum with more weight on recent observations.  Compared
    to a simple sum, the EWMA reacts faster to changes in trend but still
    smooths noisy price fluctuations.  The output is lagged by one period
    to enforce no look‑ahead bias.

    Parameters
    ----------
    prices : DataFrame
        Asset price series with datetime index and asset columns.
    span : int, default 20
        Span parameter for the EWMA.  Larger values result in a slower
        decay (more historical influence), while smaller values emphasise
        recent returns.

    Returns
    -------
    DataFrame
        Lagged EWMA momentum scores.
    """
    rets = prices.pct_change()
    # exponential moving average of returns; adjust=False for simple decay
    ewma = rets.ewm(span=span, adjust=False).mean()
    return ewma.shift(1)


def moving_average_crossover_signal(
    prices: pd.DataFrame,
    ma_short: int = 10,
    ma_long: int = 50,
) -> pd.DataFrame:
    """Compute a moving average crossover signal.

    The signal is the difference between a short moving average and a long
    moving average for each asset.  A positive difference indicates the
    short‑term trend is above the long‑term trend (bullish), while a
    negative difference indicates a bearish configuration.  This
    continuous difference is used as a cross‑sectional score and is
    lagged by one period to avoid look‑ahead bias.

    Parameters
    ----------
    prices : DataFrame
        Asset price series with datetime index and asset columns.
    ma_short : int, default 10
        Window length for the short moving average.
    ma_long : int, default 50
        Window length for the long moving average.  Must be greater than
        ``ma_short`` to make sense; no error is raised but the result will
        be trivial if ``ma_long`` <= ``ma_short``.

    Returns
    -------
    DataFrame
        Lagged moving average crossover scores.
    """
    # compute moving averages
    ma_s = prices.rolling(ma_short).mean()
    ma_l = prices.rolling(ma_long).mean()
    diff = ma_s - ma_l
    return diff.shift(1)
