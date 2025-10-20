"""
Signal generation functions for time‑series alpha strategies.

This module contains a collection of cross‑sectional signals that can be used to
generate tradable rankings from price series.  All signals are lagged by one
period to avoid look‑ahead bias.

Available signals:

* ``momentum_signal`` – sums trailing returns over a lookback window.
* ``mean_reversion_signal`` – negative of trailing returns, to capture mean
  reverting behaviour.
* ``arima_signal`` – fits an ARIMA model to returns and uses one‑step ahead
  forecasts as the signal (experimental; may be slow on large universes).
* ``volatility_signal`` – negative of rolling squared returns, i.e. long low
  volatility and short high volatility names.

All functions return a pandas ``DataFrame`` aligned to the input price index
with a one‑period shift applied so that the signal at time ``t`` uses only
information up to ``t‑1``.

"""

from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # type: ignore

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
