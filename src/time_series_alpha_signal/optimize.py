from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .evaluation import cross_validate_sharpe
from .backtest import backtest


def optimize_ewma(
    prices: pd.DataFrame,
    spans: Iterable[int],
    n_splits: int = 5,
    cost_bps: float = 10.0,
    max_gross: float = 1.0,
    seed: int = 42,
) -> Dict[str, object]:
    """Search for the EWMA span that maximises out‑of‑sample Sharpe.

    For each span in ``spans`` this helper runs purged k‑fold
    cross‑validation on the ``ewma_momentum`` signal and computes the
    average Sharpe ratio across folds.  The span with the highest
    average Sharpe is returned along with a summary of all spans.

    Parameters
    ----------
    prices : DataFrame
        Price series indexed by datetime with asset columns.
    spans : iterable of int
        Candidate EWMA spans to evaluate.
    n_splits : int, default 5
        Number of cross‑validation folds.
    cost_bps : float, default 10.0
        Transaction cost in basis points.
    max_gross : float, default 1.0
        Gross exposure limit for the portfolio.
    seed : int, default 42
        Random seed forwarded to the backtest (currently unused).

    Returns
    -------
    dict
        Dictionary with keys ``best_span``, ``results`` and
        ``metrics``.  ``results`` maps each span to its list of
        fold Sharpe ratios and mean/median.  ``metrics`` contains the
        Sharpe, CAGR, max drawdown and average turnover of the final
        backtest with the best span.
    """
    prices = prices.sort_index()
    results: Dict[int, Dict[str, object]] = {}
    best_span: int | None = None
    best_mean: float = -np.inf
    # iterate over candidate spans
    for span in spans:
        sharpe_values = cross_validate_sharpe(
            prices=prices,
            signal_type="ewma_momentum",
            max_gross=max_gross,
            cost_bps=cost_bps,
            n_splits=n_splits,
            ewma_span=span,
        )
        if len(sharpe_values) == 0:
            mean_sr = float("nan")
        else:
            mean_sr = float(np.nanmean(sharpe_values))
        results[span] = {
            "sharpe_folds": sharpe_values,
            "mean_sharpe": mean_sr,
            "median_sharpe": float(np.nanmedian(sharpe_values)) if sharpe_values else float("nan"),
        }
        if mean_sr > best_mean:
            best_mean = mean_sr
            best_span = span
    if best_span is None:
        raise RuntimeError("No valid span produced a Sharpe ratio")
    # run a final backtest on full data using best_span
    backtest_out = backtest(
        prices=prices,
        signal_type="ewma_momentum",
        ewma_span=best_span,
        max_gross=max_gross,
        cost_bps=cost_bps,
        seed=seed,
    )
    metrics = backtest_out["metrics"]
    return {
        "best_span": best_span,
        "results": results,
        "metrics": metrics,
    }

