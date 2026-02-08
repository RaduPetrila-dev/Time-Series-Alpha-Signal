"""Parameter optimisation and portfolio constraint helpers.

This module provides:

* :func:`enforce_leverage` -- cap daily gross exposure on a weight
  matrix (used by the backtesting pipeline).
* :func:`optimize_parameter` -- generic grid search over a single
  signal parameter using purged cross-validated Sharpe ratios.
* :func:`optimize_ewma` -- convenience wrapper for EWMA span search
  (calls :func:`optimize_parameter` internally).

All optimisation is performed via **out-of-sample** evaluation using
purged k-fold CV to guard against overfitting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest
from .evaluation import cross_validate_sharpe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leverage constraint
# ---------------------------------------------------------------------------

def enforce_leverage(
    weights: pd.DataFrame,
    max_leverage: float,
) -> pd.DataFrame:
    """Scale daily weights so gross exposure does not exceed *max_leverage*.

    On days where the sum of absolute weights exceeds *max_leverage*,
    all weights are proportionally scaled down.  Days already within
    the limit are unchanged.

    Parameters
    ----------
    weights : DataFrame
        Daily portfolio weights (assets in columns).
    max_leverage : float
        Maximum allowed gross exposure (sum of absolute weights).

    Returns
    -------
    DataFrame
        Adjusted weights.

    Raises
    ------
    ValueError
        If *max_leverage* <= 0.
    """
    if max_leverage <= 0:
        raise ValueError(f"max_leverage must be > 0, got {max_leverage}")

    gross = weights.abs().sum(axis=1)
    scale = np.minimum(max_leverage / gross.replace(0, np.inf), 1.0)
    return weights.mul(scale, axis=0)

# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationResult:
    """Structured result from parameter optimisation.

    Attributes
    ----------
    best_value : Any
        The parameter value that maximised mean OOS Sharpe.
    best_mean_sharpe : float
        Mean cross-validated Sharpe at the best parameter.
    param_name : str
        Name of the parameter that was searched.
    grid_results : dict
        Maps each candidate value to a dict with ``sharpe_folds``,
        ``mean_sharpe``, and ``median_sharpe``.
    final_metrics : dict[str, Any]
        Full backtest metrics from running the best parameter on the
        entire dataset.
    """

    best_value: Any
    best_mean_sharpe: float
    param_name: str
    grid_results: dict[Any, dict[str, Any]]
    final_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "best_value": self.best_value,
            "best_mean_sharpe": self.best_mean_sharpe,
            "param_name": self.param_name,
            "grid_results": {
                str(k): v for k, v in self.grid_results.items()
            },
            "final_metrics": self.final_metrics,
        }

# ---------------------------------------------------------------------------
# Generic parameter search
# ---------------------------------------------------------------------------

def optimize_parameter(
    prices: pd.DataFrame,
    signal_type: str,
    param_name: str,
    candidates: Iterable[Any],
    n_splits: int = 5,
    embargo_pct: float = 0.0,
    cost_bps: float = 10.0,
    max_gross: float = 1.0,
    rebalance: str = "daily",
    impact_model: str = "proportional",
    run_final_backtest: bool = True,
    **fixed_kwargs: Any,
) -> OptimizationResult:
    """Grid-search a single parameter to maximise OOS Sharpe.

    For each value in *candidates*, runs purged k-fold CV on the
    specified signal and records the Sharpe distribution.  The value
    with the highest mean Sharpe is selected.  Optionally, a full
    backtest is run with the best parameter on the entire dataset.

    Parameters
    ----------
    prices : DataFrame
        Price series.
    signal_type : str
        Signal name (must be registered in the signal registry).
    param_name : str
        Name of the parameter to search (e.g. ``"lookback"``,
        ``"ewma_span"``, ``"ma_short"``).
    candidates : iterable
        Values to evaluate for *param_name*.
    n_splits : int, default 5
        Number of CV folds.
    embargo_pct : float, default 0.0
        Embargo fraction between folds.
    cost_bps : float, default 10.0
        Transaction cost in basis points.
    max_gross : float, default 1.0
        Gross exposure limit.
    rebalance : str, default "daily"
        Rebalance frequency.
    impact_model : str, default "proportional"
        Cost model.
    run_final_backtest : bool, default True
        Run a full-sample backtest with the best parameter.
    **fixed_kwargs
        Additional signal kwargs that remain constant across the
        search (e.g. ``vol_window=20``).

    Returns
    -------
    OptimizationResult
        Structured result with the best parameter, grid results,
        and optional final backtest metrics.

    Raises
    ------
    RuntimeError
        If no candidate produces a valid Sharpe ratio.
    """
    prices = prices.sort_index()
    candidates_list = list(candidates)

    logger.info(
        "Optimising '%s' for signal '%s': %d candidates.",
        param_name,
        signal_type,
        len(candidates_list),
    )

    grid_results: dict[Any, dict[str, Any]] = {}
    best_value: Any = None
    best_mean: float = -np.inf

    for value in candidates_list:
        cv_kwargs: dict[str, Any] = {
            param_name: value,
            **fixed_kwargs,
        }

        sharpe_values = cross_validate_sharpe(
            prices=prices,
            signal_type=signal_type,
            max_gross=max_gross,
            cost_bps=cost_bps,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            rebalance=rebalance,
            impact_model=impact_model,
            **cv_kwargs,
        )

        if len(sharpe_values) == 0:
            mean_sr = float("nan")
            median_sr = float("nan")
        else:
            mean_sr = float(np.nanmean(sharpe_values))
            median_sr = float(np.nanmedian(sharpe_values))

        grid_results[value] = {
            "sharpe_folds": sharpe_values,
            "mean_sharpe": mean_sr,
            "median_sharpe": median_sr,
        }

        logger.debug(
            "  %s=%s: mean_sharpe=%.3f, median_sharpe=%.3f (%d folds)",
            param_name,
            value,
            mean_sr,
            median_sr,
            len(sharpe_values),
        )

        if not np.isnan(mean_sr) and mean_sr > best_mean:
            best_mean = mean_sr
            best_value = value

    if best_value is None:
        raise RuntimeError(
            f"No candidate for '{param_name}' produced a valid Sharpe ratio."
        )

    logger.info(
        "Best %s=%s with mean Sharpe=%.3f.",
        param_name,
        best_value,
        best_mean,
    )

    # -- Optional full-sample backtest with best parameter ---------------
    final_metrics: dict[str, Any] = {}
    if run_final_backtest:
        bt_kwargs: dict[str, Any] = {
            param_name: best_value,
            **fixed_kwargs,
        }
        result: BacktestResult = backtest(
            prices=prices,
            signal_type=signal_type,
            max_gross=max_gross,
            cost_bps=cost_bps,
            rebalance=rebalance,
            impact_model=impact_model,
            **bt_kwargs,
        )
        final_metrics = result.metrics

    return OptimizationResult(
        best_value=best_value,
        best_mean_sharpe=best_mean,
        param_name=param_name,
        grid_results=grid_results,
        final_metrics=final_metrics,
    )

# ---------------------------------------------------------------------------
# EWMA convenience wrapper
# ---------------------------------------------------------------------------

def optimize_ewma(
    prices: pd.DataFrame,
    spans: Iterable[int],
    n_splits: int = 5,
    cost_bps: float = 10.0,
    max_gross: float = 1.0,
    rebalance: str = "daily",
    impact_model: str = "proportional",
) -> dict[str, Any]:
    """Search for the EWMA span that maximises OOS Sharpe.

    Convenience wrapper around :func:`optimize_parameter` for the
    ``ewma_momentum`` signal.

    Parameters
    ----------
    prices : DataFrame
        Price series.
    spans : iterable of int
        Candidate EWMA spans.
    n_splits : int, default 5
        CV folds.
    cost_bps : float, default 10.0
        Transaction cost in bps.
    max_gross : float, default 1.0
        Gross exposure limit.
    rebalance : str, default "daily"
        Rebalance frequency.
    impact_model : str, default "proportional"
        Cost model.

    Returns
    -------
    dict
        Keys: ``best_span``, ``results``, ``metrics``.
    """
    opt = optimize_parameter(
        prices=prices,
        signal_type="ewma_momentum",
        param_name="ewma_span",
        candidates=spans,
        n_splits=n_splits,
        cost_bps=cost_bps,
        max_gross=max_gross,
        rebalance=rebalance,
        impact_model=impact_model,
    )

    return {
        "best_span": opt.best_value,
        "results": opt.grid_results,
        "metrics": opt.final_metrics,
    }
