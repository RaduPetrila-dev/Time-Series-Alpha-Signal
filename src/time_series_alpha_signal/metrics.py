from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd


def annualised_sharpe(returns: pd.Series, periods: int = 252) -> float:
    """Compute the annualised Sharpe ratio of a return series.

    Parameters
    ----------
    returns : Series
        Daily return series.
    periods : int, default 252
        Number of periods per year.  For daily data this is 252.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    mu = returns.mean()
    sigma = returns.std(ddof=0)
    if sigma <= 0:
        return 0.0
    sr = np.sqrt(periods) * mu / sigma
    return float(sr)


def newey_west_tstat(returns: pd.Series, lags: int | None = None) -> float:
    """Compute the Newey–West adjusted t‑statistic for the mean of returns.

    The truncation lag ``lags`` can be provided manually or is set by
    the rule of thumb ``0.75 * n**(1/3)``【819955570341273†L218-L246】.  The t‑statistic
    equals the sample mean divided by the Newey–West standard error.

    Parameters
    ----------
    returns : Series
        Return series to evaluate.
    lags : int, optional
        Number of autocovariance lags to include.  If ``None`` a rule of
        thumb is used.

    Returns
    -------
    float
        Newey–West t‑statistic.
    """
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 2:
        return float("nan")
    if lags is None:
        lags = int(0.75 * n ** (1 / 3))
    # demean the series
    r_demeaned = r - r.mean()
    gamma0 = np.dot(r_demeaned, r_demeaned) / n
    var = gamma0
    for k in range(1, lags + 1):
        gamma_k = np.dot(r_demeaned[:-k], r_demeaned[k:]) / n
        weight = 1.0 - k / (lags + 1)
        var += 2.0 * weight * gamma_k
    se = np.sqrt(var / n)
    if se <= 0:
        return float("inf")
    t_stat = r.mean() / se
    return float(t_stat)


def block_bootstrap_ci(
    metric_func: Callable[[pd.Series], float],
    returns: pd.Series,
    block_size: int | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Estimate confidence intervals for a metric using block bootstrap.

    Parameters
    ----------
    metric_func : callable
        Function that computes a statistic from a return series (e.g.
        :func:`annualised_sharpe`).
    returns : Series
        Return series to resample.
    block_size : int, optional
        Size of each contiguous block.  Defaults to ``int(sqrt(n))`` where
        ``n`` is the number of observations.  Blocks wrap around if
        necessary.
    n_bootstrap : int, default 1000
        Number of bootstrap replications.
    alpha : float, default 0.05
        Significance level for the two‑sided confidence interval.

    Returns
    -------
    (float, float)
        Lower and upper bounds of the bootstrap confidence interval.
    """
    r = returns.dropna().astype(float)
    n = len(r)
    if n == 0:
        raise ValueError("returns must contain at least one observation")
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))
    # precompute blocks; wrap around at the end of the series
    blocks = []
    for start in range(n):
        end = start + block_size
        if end <= n:
            blocks.append(r.iloc[start:end].values)
        else:
            # wrap around by concatenating from beginning
            wrap = np.concatenate([r.iloc[start:].values, r.iloc[: end - n].values])
            blocks.append(wrap)
    metrics = []
    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        sampled = []
        # draw enough blocks to exceed n observations
        while len(sampled) < n:
            idx = rng.integers(0, len(blocks))
            sampled.extend(blocks[idx])
        sampled_series = pd.Series(sampled[:n], index=r.index)
        metrics.append(metric_func(sampled_series))
    lower = np.percentile(metrics, 100 * alpha / 2)
    upper = np.percentile(metrics, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> float:
    """Compute a simple deflated Sharpe ratio.

    The deflated Sharpe ratio penalises the observed Sharpe ratio for
    multiple testing and non‑normality【908685318920237†L142-L149】.  This
    implementation applies a conservative adjustment proportional to
    ``sqrt(n_trials)`` to reduce the likelihood of false discoveries.

    Parameters
    ----------
    returns : Series
        Return series.
    n_trials : int
        Number of independent trials (e.g. number of strategies or
        parameter combinations tested).

    Returns
    -------
    float
        Deflated Sharpe ratio.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    sr = annualised_sharpe(returns)
    n = returns.dropna().shape[0]
    # approximate the standard error of the Sharpe ratio
    if n <= 1:
        return sr
    se_sr = np.sqrt((1 + 0.5 * sr ** 2) / (n - 1))
    # adjust Sharpe by a penalty term depending on number of trials
    penalty = se_sr * np.sqrt(n_trials)
    return float(sr - penalty)
