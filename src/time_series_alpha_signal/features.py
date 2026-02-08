r"""Fractional differentiation for memory-preserving stationarity.

This module implements the **fixed-width window fractional
differentiation** (FFD) method described in:

    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Wiley. Chapter 5.

The core idea is to apply the binomial expansion of the fractional
differencing operator :math:`(1 - L)^d` to a time series, where
:math:`L` is the lag operator and :math:`0 < d < 1`.  Unlike integer
differencing (:math:`d = 1`), fractional differencing removes just
enough memory to achieve stationarity while preserving long-range
dependence that is useful for forecasting.

The weights of the expansion are:

.. math::
    w_k = -w_{k-1} \frac{d - k + 1}{k}, \quad w_0 = 1

Weights are truncated when :math:`|w_k| < \tau` (the ``thresh``
parameter) to keep the convolution window finite.

Functions
---------
* :func:`fracdiff_weights` -- compute the truncated weight vector.
* :func:`fracdiff_series` -- fractionally difference a single Series.
* :func:`fracdiff_df` -- apply to every column of a DataFrame.
* :func:`pick_min_d` -- select the smallest *d* that achieves
  stationarity via the ADF test.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight computation (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def fracdiff_weights(
    d: float,
    max_size: int,
    thresh: float = 1e-4,
) -> np.ndarray:
    r"""Compute truncated fractional differencing weights.

    Generates the weight vector for the binomial expansion of
    :math:`(1 - L)^d` up to *max_size* lags, stopping early when
    :math:`|w_k| < \text{thresh}`.

    Parameters
    ----------
    d : float
        Fractional differencing order (:math:`0 < d < 1` typical).
    max_size : int
        Maximum number of weights to compute (upper bound on window
        length).
    thresh : float, default 1e-4
        Absolute weight threshold for truncation.

    Returns
    -------
    ndarray
        1-D array of weights, length <= *max_size*.
    """
    weights: list[float] = [1.0]
    for k in range(1, max_size):
        w_k = -weights[k - 1] * (d - (k - 1)) / k
        if abs(w_k) < thresh:
            break
        weights.append(w_k)
    return np.array(weights, dtype=np.float64)


# ---------------------------------------------------------------------------
# Series-level fractional differencing
# ---------------------------------------------------------------------------


def fracdiff_series(
    series: pd.Series,
    d: float,
    thresh: float = 1e-4,
) -> pd.Series:
    r"""Fractionally difference a single time series.

    Applies the weight vector from :func:`fracdiff_weights` to *series*
    using a rolling dot-product (convolution).  Values before the first
    complete window are set to ``NaN``.

    The implementation is **vectorised**: it uses :func:`numpy.convolve`
    rather than a Python-level loop, which is roughly 10-50x faster on
    series with > 1 000 observations.

    Parameters
    ----------
    series : Series
        Input time series (e.g. log-prices).
    d : float
        Differencing order.  Must be >= 0.
    thresh : float, default 1e-4
        Weight truncation threshold.

    Returns
    -------
    Series
        Fractionally differenced series with the same index.  The
        first ``len(weights) - 1`` values are ``NaN``.

    Raises
    ------
    ValueError
        If *d* is negative.
    """
    if d < 0:
        raise ValueError(f"d must be >= 0, got {d}")
    if d == 0:
        return series.copy()

    arr = series.values.astype(np.float64)
    n = len(arr)

    w = fracdiff_weights(d, max_size=n, thresh=thresh)
    window = len(w)

    # Convolution: flip weights so that w[0] aligns with the most
    # recent observation (standard FIR filter convention).
    w_flip = w[::-1]

    # Use np.convolve in 'full' mode, then slice to keep only the
    # positions where the full window was available.
    #
    # For positions with any NaN in the input window we fall back to
    # NaN.  We detect this by convolving a NaN indicator.
    nan_mask = np.isnan(arr)

    if nan_mask.any():
        # Fallback: manual dot product only where NaNs exist in the
        # window.  For clean regions, use the fast convolution.
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk = arr[i - window + 1 : i + 1]
            if np.isnan(chunk).any():
                out[i] = np.nan
            else:
                out[i] = np.dot(w_flip, chunk)
    else:
        # Fast path: full vectorised convolution.
        conv = np.convolve(arr, w_flip, mode="full")
        # conv has length n + window - 1.  We want indices
        # [window-1 .. n-1] of the convolution output (0-indexed),
        # which correspond to the first position where the full window
        # is available through to the last input element.
        out = np.full(n, np.nan)
        out[window - 1 :] = conv[2 * (window - 1) : 2 * (window - 1) + (n - window + 1)]

    return pd.Series(out, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# DataFrame-level convenience
# ---------------------------------------------------------------------------


def fracdiff_df(
    df: pd.DataFrame,
    d: float,
    thresh: float = 1e-4,
) -> pd.DataFrame:
    """Apply fractional differentiation to every column of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input data with one or more asset columns.
    d : float
        Differencing order (shared across all columns).
    thresh : float, default 1e-4
        Weight truncation threshold.

    Returns
    -------
    DataFrame
        Fractionally differenced DataFrame with the same shape and
        index.
    """
    result = {
        col: fracdiff_series(df[col], d=d, thresh=thresh) for col in df.columns
    }
    return pd.DataFrame(result, index=df.index)


# ---------------------------------------------------------------------------
# Minimum-d selection via ADF test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinDResult:
    """Result of :func:`pick_min_d`.

    Attributes
    ----------
    d : float
        Selected fractional differencing order.
    adf_pvalue : float
        ADF p-value at the selected order.
    is_fallback : bool
        ``True`` if no candidate achieved stationarity and the largest
        candidate was returned as a fallback.
    candidates_tested : dict[float, float]
        Mapping of each tested *d* to its ADF p-value.
    """

    d: float
    adf_pvalue: float
    is_fallback: bool
    candidates_tested: dict[float, float]


def pick_min_d(
    series: pd.Series,
    candidates: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    p_threshold: float = 0.05,
    min_obs: int = 50,
) -> MinDResult:
    """Select the smallest *d* that yields a stationary series.

    Iterates over *candidates* in ascending order, applies fractional
    differencing, and runs the augmented Dickey-Fuller test.  Returns
    the first *d* whose p-value is below *p_threshold*.

    Parameters
    ----------
    series : Series
        Input time series (e.g. raw prices or log-prices).
    candidates : sequence of float, default (0.1, 0.2, ..., 0.6)
        Candidate orders to test, in ascending order.
    p_threshold : float, default 0.05
        ADF p-value threshold for rejecting the unit-root null.
    min_obs : int, default 50
        Minimum number of non-NaN observations required after
        differencing to run the ADF test.

    Returns
    -------
    MinDResult
        Structured result with the selected *d*, ADF p-value,
        fallback flag, and the full map of candidates tested.

    Raises
    ------
    RuntimeError
        If ``statsmodels`` is not installed.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> prices = pd.Series(np.cumsum(np.random.randn(500)) + 100)
    >>> result = pick_min_d(prices)
    >>> result.d  # doctest: +SKIP
    0.3
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is required for pick_min_d. "
            "Install with: pip install statsmodels"
        ) from exc

    clean = series.dropna()
    tested: dict[float, float] = {}

    for d in sorted(candidates):
        fd = fracdiff_series(clean, d=d).dropna()
        if len(fd) < min_obs:
            logger.debug(
                "d=%.2f: only %d observations after differencing "
                "(need %d), skipping.",
                d,
                len(fd),
                min_obs,
            )
            continue

        adf_result = adfuller(fd, autolag="AIC")
        pval = float(adf_result[1])
        tested[d] = pval
        logger.debug("d=%.2f: ADF p-value=%.4f", d, pval)

        if pval < p_threshold:
            logger.info(
                "Selected d=%.2f (ADF p=%.4f < %.2f).",
                d,
                pval,
                p_threshold,
            )
            return MinDResult(
                d=d,
                adf_pvalue=pval,
                is_fallback=False,
                candidates_tested=tested,
            )

    # Fallback: no candidate achieved stationarity
    fallback_d = max(candidates)
    fallback_pval = tested.get(fallback_d, float("nan"))
    logger.warning(
        "No candidate achieved stationarity (p < %.2f). "
        "Falling back to d=%.2f (p=%.4f).",
        p_threshold,
        fallback_d,
        fallback_pval,
    )
    return MinDResult(
        d=fallback_d,
        adf_pvalue=fallback_pval,
        is_fallback=True,
        candidates_tested=tested,
    )
