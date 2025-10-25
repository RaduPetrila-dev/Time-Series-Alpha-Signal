from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _fracdiff_weights(d: float, size: int) -> np.ndarray:
    """Compute fractional differencing weights of length ``size``.

    The weights correspond to the binomial expansion of ``(1 − L)**d``,
    where ``L`` is the lag operator.  See López de Prado for details.

    Parameters
    ----------
    d : float
        Fractional differencing order.
    size : int
        Number of weights to compute.

    Returns
    -------
    ndarray
        Array of fractional differencing weights.
    """
    w = np.ones(size)
    for k in range(1, size):
        w[k] = -w[k - 1] * (d - (k - 1)) / k
    return w


def fracdiff_series(
    series: pd.Series,
    d: float,
    thresh: float = 1e-4,
) -> pd.Series:
    """Compute the fractionally differenced version of a series.

    Parameters
    ----------
    series : Series
        Input time series to be fractionally differentiated.
    d : float
        Differencing order.  Values between 0 and 1 typically reduce
        memory while preserving stationarity.【678063069448557†L43-L63】
    thresh : float, default 1e-4
        Threshold for truncating small weights.  When the absolute
        weight at lag ``k`` is below this value the remaining lags are
        ignored.

    Returns
    -------
    Series
        Fractionally differenced series with the same index as the
        input.  Values before the first valid observation are ``NaN``.
    """
    if d < 0:
        raise ValueError("d must be non‑negative")
    # compute weights until they fall below the threshold
    max_size = len(series)
    weights: List[float] = []
    k = 0
    while True:
        if k == 0:
            weights.append(1.0)
        else:
            weights.append(-weights[k - 1] * (d - (k - 1)) / k)
        if abs(weights[-1]) < thresh or k >= max_size - 1:
            break
        k += 1
    w = np.array(weights)
    # apply convolution of weights with values
    out = np.zeros_like(series, dtype=float)
    arr = series.values
    for i in range(len(w), len(arr) + 1):
        window = arr[i - len(w) : i]
        if np.any(pd.isna(window)):
            out[i - 1] = np.nan
        else:
            out[i - 1] = np.dot(w[::-1], window)
    return pd.Series(out, index=series.index)


def pick_min_d(
    series: pd.Series,
    candidates: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6),
    p_threshold: float = 0.05,
) -> float:
    """Select the smallest differencing order that yields stationarity.

    This helper iterates over a set of candidate fractional orders and
    computes the augmented Dickey–Fuller (ADF) test on the
    fractionally differenced series.  The first candidate whose ADF
    p‑value is below ``p_threshold`` is returned.  If no candidate
    satisfies the criterion the largest candidate is returned.  The
    implementation relies on ``statsmodels.tsa.stattools.adfuller``; if
    ``statsmodels`` is not installed a ``RuntimeError`` is raised.

    Parameters
    ----------
    series : Series
        Input time series (e.g. price series) to analyse.
    candidates : tuple of float, default (0.2, 0.3, 0.4, 0.5, 0.6)
        Candidate fractional orders to test.  Values should be between
        0 and 1.
    p_threshold : float, default 0.05
        P‑value threshold for rejecting the null hypothesis of a unit
        root (i.e. the series is stationary if p < threshold).

    Returns
    -------
    float
        The selected fractional order.

    Raises
    ------
    RuntimeError
        If ``statsmodels`` is not available.
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "statsmodels is required for pick_min_d. Please install it via 'pip install statsmodels'."
        ) from e
    best = None
    for d in candidates:
        fd = fracdiff_series(series.dropna(), d=d).dropna()
        if len(fd) < 50:
            continue
        adf_stat = adfuller(fd, autolag="AIC")
        pval = adf_stat[1]
        if pval < p_threshold:
            best = d
            break
    if best is not None:
        return best
    # fallback: return the largest candidate
    return max(candidates)


def fracdiff_df(
    df: pd.DataFrame,
    d: float,
    thresh: float = 1e-4,
) -> pd.DataFrame:
    """Apply fractional differentiation to each column of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input data with one or more columns.
    d : float
        Differencing order shared across all columns.
    thresh : float, default 1e-4
        Threshold for truncating small weights.

    Returns
    -------
    DataFrame
        Fractionally differenced DataFrame.
    """
    return pd.concat({col: fracdiff_series(df[col], d=d, thresh=thresh) for col in df}, axis=1)
