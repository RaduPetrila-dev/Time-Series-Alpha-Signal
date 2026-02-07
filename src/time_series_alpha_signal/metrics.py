r"""Statistical metrics for strategy evaluation.

This module provides:

* :func:`annualised_sharpe` -- annualised Sharpe ratio.
* :func:`newey_west_tstat` -- HAC-robust t-statistic for the mean
  return (Newey & West, 1987).
* :func:`block_bootstrap_ci` -- block bootstrap confidence intervals
  for any scalar metric.
* :func:`deflated_sharpe_ratio` -- the probabilistic Sharpe ratio
  (PSR) adjusted for multiple testing (Bailey & Lopez de Prado, 2014).

References
----------
.. [1] Newey, W. K. and West, K. D. (1987). "A Simple, Positive
   Semi-definite, Heteroskedasticity and Autocorrelation Consistent
   Covariance Matrix." *Econometrica*, 55(3), 703-708.

.. [2] Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated
   Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
   and Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Annualised Sharpe ratio
# ---------------------------------------------------------------------------

def annualised_sharpe(
    returns: pd.Series,
    periods: int = 252,
) -> float:
    r"""Compute the annualised Sharpe ratio.

    .. math::
        \text{SR} = \sqrt{N} \; \frac{\bar{r}}{\hat\sigma}

    where :math:`N` is *periods*, :math:`\bar{r}` is the sample mean,
    and :math:`\hat\sigma` is the sample standard deviation
    (``ddof=1``).

    Parameters
    ----------
    returns : Series
        Return series (daily, weekly, etc.).
    periods : int, default 252
        Annualisation factor.

    Returns
    -------
    float
        Annualised Sharpe ratio, or 0.0 if volatility is zero.
    """
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    sigma = r.std(ddof=1)
    if sigma <= 0:
        return 0.0
    return float(np.sqrt(periods) * r.mean() / sigma)

# ---------------------------------------------------------------------------
# Newey-West t-statistic
# ---------------------------------------------------------------------------

def newey_west_tstat(
    returns: pd.Series,
    lags: int | None = None,
) -> float:
    r"""Compute the Newey-West HAC t-statistic for the mean return.

    Uses the Bartlett kernel:

    .. math::
        \hat V = \gamma_0 + 2 \sum_{k=1}^{L}
            \Bigl(1 - \frac{k}{L+1}\Bigr) \gamma_k

    where :math:`\gamma_k` is the sample autocovariance at lag *k* and
    *L* is the truncation lag.

    The t-statistic is :math:`\bar{r} / \text{SE}` where
    :math:`\text{SE} = \sqrt{\hat V / n}`.

    Parameters
    ----------
    returns : Series
        Return series.
    lags : int, optional
        Truncation lag.  Defaults to the Newey-West rule of thumb
        :math:`\lfloor 0.75 \, n^{1/3} \rfloor`.

    Returns
    -------
    float
        HAC t-statistic, or ``nan`` if fewer than 2 observations.
    """
    r = returns.dropna().astype(np.float64)
    n = len(r)
    if n < 2:
        return float("nan")

    if lags is None:
        lags = int(0.75 * n ** (1 / 3))

    mean = r.mean()
    r_dm = (r - mean).values

    # Autocovariances via vectorised dot products
    gamma0 = float(np.dot(r_dm, r_dm) / n)
    nw_var = gamma0
    for k in range(1, lags + 1):
        gamma_k = float(np.dot(r_dm[:-k], r_dm[k:]) / n)
        weight = 1.0 - k / (lags + 1)
        nw_var += 2.0 * weight * gamma_k

    # Guard against numerical issues
    if nw_var <= 0:
        logger.warning(
            "Newey-West variance is non-positive (%.2e); returning inf.",
            nw_var,
        )
        return float("inf")

    se = np.sqrt(nw_var / n)
    return float(mean / se)

# ---------------------------------------------------------------------------
# Block bootstrap confidence intervals
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    metric_func: Callable[[pd.Series], float],
    returns: pd.Series,
    block_size: int | None = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    r"""Estimate a confidence interval via circular block bootstrap.

    Blocks of length *block_size* are drawn uniformly at random (with
    wrap-around) and concatenated to form a resampled series of the
    same length as the original.  *metric_func* is evaluated on each
    replicate to produce the bootstrap distribution.

    Parameters
    ----------
    metric_func : callable
        ``f(returns: Series) -> float``.
    returns : Series
        Return series to resample.
    block_size : int, optional
        Block length.  Defaults to :math:`\lfloor\sqrt{n}\rfloor`.
    n_bootstrap : int, default 1000
        Number of bootstrap replications.
    alpha : float, default 0.05
        Significance level for the two-sided CI.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    (lower, upper) : tuple of float
        Bootstrap confidence interval bounds.

    Raises
    ------
    ValueError
        If *returns* is empty after dropping NaNs.
    """
    r = returns.dropna().astype(np.float64)
    n = len(r)
    if n == 0:
        raise ValueError("returns must contain at least one observation.")

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    arr = r.values
    rng = np.random.default_rng(seed)

    # Pre-compute circular blocks as a 2-D array for fast indexing.
    # block_starts[i] gives the start index; we use modular indexing
    # to handle wrap-around.
    n_blocks_needed = int(np.ceil(n / block_size))

    metrics = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks_needed)
        # Build resampled array using modular indexing
        indices = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
        sampled = pd.Series(arr[indices], index=r.index)
        metrics[b] = metric_func(sampled)

    lower = float(np.percentile(metrics, 100 * alpha / 2))
    upper = float(np.percentile(metrics, 100 * (1 - alpha / 2)))

    logger.debug(
        "Block bootstrap (n=%d, blocks=%d, B=%d): %.1f%% CI = [%.4f, %.4f]",
        n,
        block_size,
        n_bootstrap,
        100 * (1 - alpha),
        lower,
        upper,
    )

    return lower, upper

# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (PSR*)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    periods: int = 252,
) -> float:
    r"""Compute the deflated Sharpe ratio (PSR*).

    Implements the probabilistic Sharpe ratio adjusted for multiple
    testing from Bailey & Lopez de Prado (2014).

    The observed Sharpe ratio is tested against the expected maximum
    Sharpe under the null hypothesis that all *n_trials* strategies
    have zero true Sharpe.  The expected maximum is approximated by:

    .. math::
        \text{SR}^* \approx \sqrt{V[\hat{\text{SR}}]}
        \Bigl(
            (1 - \gamma)\,\Phi^{-1}\!\bigl(1 - \tfrac{1}{N}\bigr)
            + \gamma\,\Phi^{-1}\!\bigl(1 - \tfrac{1}{N}\,e^{-1}\bigr)
        \Bigr)

    where :math:`\gamma \approx 0.5772` is the Euler-Mascheroni
    constant, :math:`N` is *n_trials*, and :math:`V[\hat{\text{SR}}]`
    accounts for skewness and kurtosis.

    The deflated Sharpe ratio is the probability that the observed
    Sharpe exceeds this benchmark:

    .. math::
        \text{PSR}^* = \Phi\!\Biggl(
            \frac{(\hat{\text{SR}} - \text{SR}^*)\sqrt{n-1}}
            {\sqrt{1 - \hat\gamma_3\,\hat{\text{SR}}
            + \tfrac{\hat\gamma_4 - 1}{4}\,\hat{\text{SR}}^2}}
        \Biggr)

    Parameters
    ----------
    returns : Series
        Return series.
    n_trials : int
        Number of independent strategies or parameter sets tested.
    periods : int, default 252
        Annualisation factor.

    Returns
    -------
    float
        PSR* value in [0, 1].  Values above 0.95 indicate statistical
        significance at the 5% level.

    Raises
    ------
    ValueError
        If *n_trials* < 1.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    r = returns.dropna().astype(np.float64)
    n = len(r)
    if n < 3:
        return 0.0

    sr = annualised_sharpe(r, periods=periods)

    # Non-annualised Sharpe (per-period) for moment adjustments
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma <= 0:
        return 0.0
    sr_per_period = mu / sigma

    # Higher moments
    skew = float(sp_stats.skew(r, bias=False))
    kurt = float(sp_stats.kurtosis(r, bias=False, fisher=True))  # excess

    # Variance of the Sharpe ratio estimator (Lo, 2002)
    var_sr = (1 - skew * sr_per_period + ((kurt) / 4) * sr_per_period**2) / (n - 1)

    if var_sr <= 0:
        return 0.0

    se_sr = np.sqrt(var_sr)

    # Expected maximum Sharpe under the null (Euler-Mascheroni approx)
    euler_mascheroni = 0.5772156649
    if n_trials <= 1:
        sr_star = 0.0
    else:
        sr_star = (
            se_sr
            * np.sqrt(periods)
            * (
                (1 - euler_mascheroni) * sp_stats.norm.ppf(1 - 1.0 / n_trials)
                + euler_mascheroni * sp_stats.norm.ppf(1 - 1.0 / n_trials * np.exp(-1))
            )
        )

    # PSR*: probability that observed SR exceeds the benchmark
    if se_sr * np.sqrt(periods) <= 0:
        return 0.0

    z = (sr - sr_star) / (se_sr * np.sqrt(periods))
    psr = float(sp_stats.norm.cdf(z))

    logger.debug(
        "Deflated SR: observed=%.3f, benchmark=%.3f, "
        "PSR*=%.4f (n=%d, trials=%d, skew=%.2f, kurt=%.2f)",
        sr,
        sr_star,
        psr,
        n,
        n_trials,
        skew,
        kurt,
    )

    return psr
