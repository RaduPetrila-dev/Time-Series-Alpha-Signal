"""Evaluation helpers for cross-validated performance analysis.

This module provides:

* :func:`cross_validate_sharpe` -- estimate an out-of-sample Sharpe
  ratio distribution using purged k-fold or combinatorial purged CV.
* :func:`summarize_fold_metrics` -- compute a full performance summary
  (CAGR, volatility, Sharpe, drawdown, Calmar, hit-rate, win/loss,
  turnover) from a return series.

Both functions are designed to work with the outputs of the backtesting
pipeline in :mod:`time_series_alpha_signal.backtest`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestResult, backtest
from .cv import PurgedKFold, combinatorial_purged_cv, make_event_table
from .metrics import annualised_sharpe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-validated Sharpe distribution
# ---------------------------------------------------------------------------


def cross_validate_sharpe(
    prices: pd.DataFrame,
    signal_type: str = "momentum",
    lookback: int = 20,
    max_gross: float = 1.0,
    cost_bps: float = 10.0,
    n_splits: int = 5,
    embargo_pct: float = 0.0,
    combinatorial: bool = False,
    test_fold_size: int = 1,
    rebalance: str = "daily",
    impact_model: str = "proportional",
    **signal_kwargs: Any,
) -> list[float]:
    """Estimate a distribution of Sharpe ratios via purged CV.

    Runs a single backtest on the full dataset to obtain daily net
    returns, then slices the returns into purged k-fold (or CPCV)
    test windows and computes the annualised Sharpe ratio on each
    out-of-sample segment.

    Parameters
    ----------
    prices : DataFrame
        Price series indexed by datetime with asset columns.
    signal_type : str, default "momentum"
        Signal name passed to :func:`backtest`.
    lookback : int, default 20
        Lookback window for simple signals.
    max_gross : float, default 1.0
        Gross exposure limit.
    cost_bps : float, default 10.0
        Transaction cost in basis points.
    n_splits : int, default 5
        Number of CV folds.
    embargo_pct : float, default 0.0
        Embargo fraction after each test fold.
    combinatorial : bool, default False
        Use CPCV instead of standard purged k-fold.
    test_fold_size : int, default 1
        Number of folds per test set (CPCV only).
    rebalance : str, default "daily"
        Rebalance frequency passed to :func:`backtest`.
    impact_model : str, default "proportional"
        Cost model passed to :func:`backtest`.
    **signal_kwargs
        Additional keyword arguments forwarded to :func:`backtest`
        (e.g. ``vol_window``, ``ewma_span``).

    Returns
    -------
    list of float
        Annualised Sharpe ratio for each out-of-sample fold.

    Notes
    -----
    This function assumes the signal is computed from a lagged window
    and does not require fitting on the training set.  For ML-based
    signals that need a train/predict loop per fold, implement the
    loop yourself using :class:`PurgedKFold` directly.
    """
    prices = prices.sort_index()

    result: BacktestResult = backtest(
        prices=prices,
        lookback=lookback,
        max_gross=max_gross,
        cost_bps=cost_bps,
        signal_type=signal_type,
        rebalance=rebalance,
        impact_model=impact_model,
        **signal_kwargs,
    )
    daily_returns: pd.Series = result.daily

    # Build event table from price index (horizon=1 for daily returns)
    events = make_event_table(prices.index, horizon=1)

    # Select CV iterator
    if combinatorial:
        cv_iter = combinatorial_purged_cv(
            events=events,
            n_splits=n_splits,
            test_fold_size=test_fold_size,
            embargo_pct=embargo_pct,
        )
    else:
        cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
        cv_iter = cv.split(events)

    sharpe_values: list[float] = []
    n_empty = 0

    for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
        test_dates = events.iloc[test_idx]["t0"]
        fold_returns = daily_returns.reindex(test_dates).dropna()

        if len(fold_returns) == 0:
            n_empty += 1
            logger.warning("Fold %d: empty test returns, skipping.", fold_idx)
            continue

        sr = annualised_sharpe(fold_returns)
        sharpe_values.append(sr)
        logger.debug(
            "Fold %d: test_days=%d, Sharpe=%.3f",
            fold_idx,
            len(fold_returns),
            sr,
        )

    if n_empty > 0:
        logger.warning(
            "%d of %d folds had empty test returns and were skipped.",
            n_empty,
            n_empty + len(sharpe_values),
        )

    logger.info(
        "CV Sharpe distribution: n=%d, mean=%.3f, std=%.3f",
        len(sharpe_values),
        float(np.mean(sharpe_values)) if sharpe_values else float("nan"),
        float(np.std(sharpe_values)) if sharpe_values else float("nan"),
    )

    return sharpe_values


# ---------------------------------------------------------------------------
# Fold-level performance summary
# ---------------------------------------------------------------------------


def summarize_fold_metrics(
    returns: pd.Series,
    weights: pd.DataFrame | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute a full performance summary for a return series.

    Parameters
    ----------
    returns : Series
        Daily net returns.
    weights : DataFrame, optional
        Portfolio weights aligned to *returns*.  Required for turnover
        computation.
    periods_per_year : int, default 252
        Trading periods per year (used for annualisation).

    Returns
    -------
    dict
        Keys: ``cagr``, ``ann_vol``, ``sharpe``, ``max_drawdown``,
        ``calmar``, ``hit_rate``, ``avg_win``, ``avg_loss``,
        ``win_loss_ratio``, ``turnover``, ``n_days``.

    Raises
    ------
    ValueError
        If *returns* is empty after dropping NaNs.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns must contain at least one observation.")

    n_days = len(r)

    # -- Equity and drawdown -----------------------------------------------
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())

    # -- CAGR --------------------------------------------------------------
    final_equity = float(equity.iloc[-1])
    if final_equity > 0 and n_days > 0:
        cagr = float(final_equity ** (periods_per_year / n_days) - 1)
    else:
        cagr = -1.0

    # -- Annualised volatility ---------------------------------------------
    ann_vol = float(np.sqrt(periods_per_year) * r.std(ddof=1))

    # -- Sharpe ratio ------------------------------------------------------
    vol = r.std(ddof=1)
    sharpe = float(np.sqrt(periods_per_year) * r.mean() / vol) if vol > 0 else 0.0

    # -- Calmar ratio (CAGR / |Max DD|, not Sharpe / |Max DD|) -------------
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # -- Hit rate ----------------------------------------------------------
    wins = r[r > 0]
    losses = r[r < 0]
    total_trades = len(wins) + len(losses)
    hit_rate = float(len(wins) / total_trades) if total_trades > 0 else 0.0

    # -- Average win / average loss ----------------------------------------
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # -- Turnover ----------------------------------------------------------
    if weights is not None:
        w = weights.reindex(r.index)
        turnover = float(w.diff().abs().sum(axis=1).mean())
    else:
        turnover = float("nan")

    return {
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "turnover": turnover,
        "n_days": n_days,
    }
