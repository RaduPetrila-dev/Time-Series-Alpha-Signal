from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import backtest
from .cv import PurgedKFold, combinatorial_purged_cv
from .metrics import annualised_sharpe


def build_event_table(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Construct a simple events table from a date index.

    Each row corresponds to one period.  The event starts at ``t0``
    and ends at the next timestamp ``t1``.  The final observation is
    dropped as it has no subsequent period.  This minimal event
    definition is sufficient for cross‑sectional strategies where
    returns at time ``t+1`` are predicted by signals computed at time
    ``t``.

    Parameters
    ----------
    index : DatetimeIndex
        Index of the price series.

    Returns
    -------
    DataFrame
        Event table with columns ``t0`` and ``t1`` indexed by event
        order.
    """
    if len(index) < 2:
        raise ValueError("index must contain at least two timestamps")
    t0 = index[:-1]
    t1 = index[1:]
    events = pd.DataFrame({"t0": t0, "t1": t1})
    return events


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
    seed: int = 42,
    **signal_kwargs: Dict[str, object],
) -> List[float]:
    """Estimate a distribution of Sharpe ratios using purged cross‑validation.

    This helper runs a single backtest on the full price dataset to
    obtain daily net returns and then slices the returns according to
    purged k‑fold splits.  For each split the annualised Sharpe ratio
    is computed on the out‑of‑sample days.  When ``combinatorial`` is
    ``True`` the combinatorial purged CV (CPCV) is used, enumerating
    all combinations of test folds of size ``test_fold_size``【95993860354040†L629-L645】.

    Parameters
    ----------
    prices : DataFrame
        Price series indexed by datetime with asset columns.
    signal_type : str, default "momentum"
        Signal name passed to :func:`~time_series_alpha_signal.backtest.backtest`.
    lookback : int, default 20
        Lookback window for simple signals.
    max_gross : float, default 1.0
        Gross exposure limit for the portfolio.
    cost_bps : float, default 10.0
        Transaction cost in basis points.
    n_splits : int, default 5
        Number of folds for cross‑validation.
    embargo_pct : float, default 0.0
        Fraction of the data to embargo after each test fold.
    combinatorial : bool, default False
        If True use combinatorial purged CV; otherwise standard
        purged k‑fold is used.
    test_fold_size : int, default 1
        Size of the test set in folds for CPCV.
    seed : int, default 42
        Random seed passed to the underlying backtest (currently
        unused but reserved for reproducibility).
    **signal_kwargs : dict
        Additional keyword arguments forwarded to
        :func:`~time_series_alpha_signal.backtest.backtest` depending on
        the selected signal type.

    Returns
    -------
    list of float
        List of annualised Sharpe ratios for each out‑of‑sample fold.

    Notes
    -----
    This function assumes that the signal computation does not rely on
    fitting a model on the training set.  For predictive models that
    require training (e.g. machine learning classifiers) you should
    train the model on the training subset within the loop and then
    generate forecasts for the test subset.
    """
    # ensure chronological ordering
    prices = prices.sort_index()
    # run full backtest once; weights and returns are computed using a
    # lagged signal and therefore do not look ahead
    out = backtest(
        prices=prices,
        lookback=lookback,
        max_gross=max_gross,
        cost_bps=cost_bps,
        seed=seed,
        signal_type=signal_type,
        **signal_kwargs,
    )
    daily_returns: pd.Series = out["daily"]
    # build minimal events table from price dates
    events = build_event_table(prices.index)
    # choose CV iterator
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
    sharpe_values: List[float] = []
    for train_idx, test_idx in cv_iter:
        test_dates = events.iloc[test_idx]["t0"]
        # slice returns to test period; drop NaNs to avoid invalid Sharpe
        fold_returns = daily_returns.reindex(test_dates).dropna()
        if len(fold_returns) == 0:
            # skip empty folds (e.g. due to missing returns at start)
            continue
        sr = annualised_sharpe(fold_returns)
        sharpe_values.append(sr)
    return sharpe_values


def summarize_fold_metrics(
    returns: pd.Series,
    weights: Optional[pd.DataFrame] = None,
    costs_bps: float = 10.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Compute a suite of performance metrics for a return series.

    Metrics include annualised CAGR, standard deviation, Sharpe ratio,
    maximum drawdown, Calmar ratio (CAGR / |Max DD|), turnover and
    average win/loss.  Turnover requires the corresponding weights.

    Parameters
    ----------
    returns : Series
        Net daily returns for the period.
    weights : DataFrame, optional
        Daily portfolio weights aligned to ``returns``.  If provided,
        turnover is computed as the average absolute change in weights.
    costs_bps : float, default 10.0
        Transaction cost used to compute gross returns from net
        returns.  If weights is None the gross returns and cost cannot
        be separated.
    periods_per_year : int, default 252
        Number of trading periods per year.

    Returns
    -------
    dict
        Dictionary of computed metrics.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns must contain at least one observation")
    days = r.shape[0]
    # cumulative equity and drawdown
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    # CAGR
    cagr = float((equity.iloc[-1] ** (periods_per_year / days) - 1))
    # annualised stdev
    stdev = float(np.sqrt(periods_per_year) * r.std(ddof=0))
    # Sharpe ratio
    sharpe = float(np.sqrt(periods_per_year) * r.mean() / (r.std(ddof=0) + 1e-12))
    # Calmar ratio
    max_dd = float(dd.min())
    calmar = sharpe / abs(max_dd) if max_dd != 0 else np.nan
    # hit rate and average win/loss
    wins = r[r > 0]
    losses = r[r < 0]
    hit_rate = float(len(wins) / len(r))
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    # turnover if weights provided
    if weights is not None:
        w = weights.reindex(r.index)
        turnover = float(w.diff().abs().sum(axis=1).mean())
    else:
        turnover = np.nan
    return {
        "cagr": cagr,
        "stdev": stdev,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "turnover": turnover,
        "n_days": days,
    }
