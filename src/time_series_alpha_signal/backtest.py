from __future__ import annotations
from typing import Any, Dict

import numpy as np
import pandas as pd

def momentum_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute a simple momentum signal based on trailing returns.

    The signal is the sum of past percentage changes over the lookback window.
    A lag of one day is applied to avoid lookahead bias.

    Parameters
    ----------
    prices : DataFrame
        Price series for each asset with datetime index and asset columns.
    lookback : int, default 20
        Number of trading days over which to compute momentum.

    Returns
    -------
    DataFrame
        Lagged momentum scores aligned to weights computation.
    """
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    return mom.shift(1)  # strict lag, no lookahead

def normalize_weights(scores: pd.DataFrame, max_gross: float = 1.0) -> pd.DataFrame:
    """Convert cross-sectional scores to portfolio weights with L1 exposure control.

    We rank scores on each day, center them to zero mean and scale so that
    the sum of absolute weights equals ``max_gross``. This produces a long/short
    portfolio with controlled gross exposure.

    Parameters
    ----------
    scores : DataFrame
        Cross-sectional signal scores.
    max_gross : float, default 1.0
        Target gross exposure (sum of absolute weights).

    Returns
    -------
    DataFrame
        Daily portfolio weights per asset.
    """
    # rank scores cross-sectionally
    ranks = scores.rank(axis=1, method="first")
    # demean ranks to ensure long/short neutrality
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    # scale to unit L1 norm and multiply by max_gross
    raw = centered.div(centered.abs().sum(axis=1), axis=0).fillna(0.0)
    return raw * max_gross

def apply_costs(returns: pd.DataFrame, weights: pd.DataFrame, bps: float = 10.0) -> pd.Series:
    """Compute portfolio returns net of proportional transaction costs.

    Costs are applied based on turnover between successive days. Turnover is
    defined as the sum of absolute changes in weights across all assets. The
    cost per day equals turnover multiplied by the basis-point cost ``bps``.

    Parameters
    ----------
    returns : DataFrame
        Daily asset returns.
    weights : DataFrame
        Portfolio weights for each day (must be aligned to returns index).
    bps : float, default 10.0
        Transaction cost in basis points (1 bps = 0.01%).

    Returns
    -------
    Series
        Daily portfolio returns net of transaction costs.
    """
    # compute daily portfolio return using weights from previous day
    gross = (weights.shift(1) * returns).sum(axis=1)
    # compute turnover (sum of abs weight changes)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost_per_day = turnover * (bps / 1e4)
    net = gross - cost_per_day
    return net

def backtest(
    prices: pd.DataFrame,
    lookback: int = 20,
    max_gross: float = 1.0,
    cost_bps: float = 10.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a simple momentum backtest with no-lookahead and transaction costs.

    Parameters
    ----------
    prices : DataFrame
        Asset price series.
    lookback : int, default 20
        Lookback period for momentum signal.
    max_gross : float, default 1.0
        Gross exposure limit.
    cost_bps : float, default 10.0
        Transaction costs in basis points.
    seed : int, default 42
        Random seed (currently unused but reserved for reproducibility).

    Returns
    -------
    dict
        A dictionary containing daily returns, cumulative equity, weights and
        summary metrics such as CAGR, Sharpe ratio and max drawdown.
    """
    # ensure chronological order
    prices = prices.sort_index()
    # compute signal and weights
    scores = momentum_signal(prices, lookback=lookback)
    weights = normalize_weights(scores, max_gross=max_gross)
    # compute daily returns (percentage change) and fill initial NaNs with zeros
    rets = prices.pct_change().fillna(0.0)
    # compute net returns after costs
    daily = apply_costs(rets, weights, bps=cost_bps)
    # compute cumulative equity curve
    equity = (1 + daily).cumprod()
    # summary metrics
    days = max(1, daily.shape[0])
    cagr = float((equity.iloc[-1] ** (252 / days) - 1))
    sharpe = float(np.sqrt(252) * daily.mean() / (daily.std() + 1e-12))
    max_dd = float((equity / equity.cummax() - 1).min())
    turnover_daily = float(weights.diff().abs().sum(axis=1).mean())
    metrics = {
        "days": int(days),
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "turnover_daily": turnover_daily,
        "max_gross": float(max_gross),
        "lookback": int(lookback),
        "cost_bps": float(cost_bps),
        "seed": int(seed),
    }
    return {"daily": daily, "equity": equity, "weights": weights, "metrics": metrics}
