from __future__ import annotations
from typing import Any, Dict, Callable, Tuple

import numpy as np
import pandas as pd

from . import signals
from . import optimizer

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
    signal_type: str = "momentum",
    arima_order: Tuple[int, int, int] = (1, 0, 1),
    max_leverage: float | None = None,
    max_drawdown: float | None = None,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
    ewma_span: int = 20,
    ma_short: int = 10,
    ma_long: int = 50,
) -> Dict[str, Any]:
    """Run a time‑series cross‑sectional backtest.

    This function generalises the basic momentum backtest by allowing
    alternative signal types, optional ARIMA order specification and
    post‑processing of weights to enforce leverage constraints.

    Parameters
    ----------
    prices : DataFrame
        Asset price series indexed by datetime.
    lookback : int, default 20
        Lookback period for simple signals (momentum, mean‑reversion and
        volatility).
    max_gross : float, default 1.0
        Gross exposure limit when normalising cross‑sectional ranks.  This
        corresponds to the sum of absolute weights before leverage scaling.
    cost_bps : float, default 10.0
        Proportional transaction cost in basis points (1 bps = 0.01%).
    seed : int, default 42
        Random seed (currently unused but reserved for reproducibility).
    signal_type : {"momentum", "mean_reversion", "arima", "volatility",
                   "vol_scaled_momentum", "regime_switch",
                   "ewma_momentum", "ma_crossover"}, default "momentum"
        Choice of signal function to use.  ``arima`` signals are based on
        ARIMA forecasts and may be slow on large datasets.  ``vol_scaled_momentum``
        scales the momentum signal by inverse realised volatility.  ``regime_switch``
        alternates between momentum and mean‑reversion depending on the
        universe‑wide volatility level.  ``ewma_momentum`` uses an exponential
        moving average of returns with span ``ewma_span``.  ``ma_crossover``
        computes the difference between a short and long moving average (windows
        ``ma_short`` and ``ma_long``) as a trend indicator.
    arima_order : tuple of int, default (1, 0, 1)
        ARIMA (p,d,q) order for the ``arima`` signal type.
    max_leverage : float, optional
        If provided, the post‑normalised weights are scaled to ensure that the
        daily gross exposure does not exceed this value.  When ``None`` the
        leverage constraint is not applied.

    max_drawdown : float, optional
        Drawdown stop as a positive fraction.  When the running equity curve
        breaches ``-max_drawdown`` (e.g. ``0.15`` corresponds to −15%), all
        subsequent returns are set to zero (the strategy goes flat).  Set
        ``None`` to disable the drawdown stop.

    vol_window : int, default 20
        Window length for computing rolling standard deviation used by the
        ``vol_scaled_momentum`` and ``regime_switch`` signals.

    vol_threshold : float, default 0.02
        Threshold for average realised volatility used by ``regime_switch`` to
        decide whether to follow momentum or mean‑reversion.  If the
        universe‑wide volatility is below this value, momentum is used; if
        above, mean‑reversion is used.

    ewma_span : int, default 20
        Span parameter for the exponential moving average used by
        ``ewma_momentum``.  Larger values produce a slower decay and
        smoother momentum estimates.

    ma_short : int, default 10
        Window length for the short moving average used by ``ma_crossover``.

    ma_long : int, default 50
        Window length for the long moving average used by ``ma_crossover``.

    Returns
    -------
    dict
        Contains the daily net returns (Series), cumulative equity (Series),
        portfolio weights (DataFrame) and a dictionary of summary metrics.
    """
    # ensure chronological order
    prices = prices.sort_index()
    # compute signal
    # choose signal function based on input
    if signal_type == "momentum":
        scores = signals.momentum_signal(prices, lookback=lookback)
    elif signal_type == "mean_reversion":
        scores = signals.mean_reversion_signal(prices, lookback=lookback)
    elif signal_type == "arima":
        scores = signals.arima_signal(prices, order=arima_order)
    elif signal_type == "volatility":
        scores = signals.volatility_signal(prices, lookback=lookback)
    elif signal_type == "vol_scaled_momentum":
        scores = signals.volatility_scaled_momentum_signal(
            prices, lookback=lookback, vol_window=vol_window
        )
    elif signal_type == "regime_switch":
        scores = signals.regime_switch_signal(
            prices, lookback=lookback, vol_window=vol_window, vol_threshold=vol_threshold
        )
    elif signal_type == "ewma_momentum":
        scores = signals.ewma_momentum_signal(prices, span=ewma_span)
    elif signal_type == "ma_crossover":
        scores = signals.moving_average_crossover_signal(
            prices, ma_short=ma_short, ma_long=ma_long
        )
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")
    # normalise to portfolio weights
    weights = normalize_weights(scores, max_gross=max_gross)
    # enforce leverage constraint if requested
    if max_leverage is not None:
        weights = optimizer.enforce_leverage(weights, max_leverage=max_leverage)
    # compute daily returns and fill initial NaNs
    rets = prices.pct_change().fillna(0.0)
    # compute net returns after costs
    daily = apply_costs(rets, weights, bps=cost_bps)

    # if a drawdown stop is specified, apply a circuit breaker: when the equity
    # curve breaches the negative drawdown threshold, subsequent daily returns
    # are set to zero (flat exposure).  This is a simple risk control that
    # avoids further losses once a maximum drawdown has been reached.
    if max_drawdown is not None:
        # compute provisional equity curve
        equity_tmp = (1 + daily).cumprod()
        dd = equity_tmp / equity_tmp.cummax() - 1.0
        breach_indices = dd[dd < -abs(max_drawdown)].index
        if len(breach_indices) > 0:
            # identify first breach date and zero out returns thereafter
            breach_start = breach_indices[0]
            daily.loc[breach_start:] = 0.0
    # recompute equity after applying drawdown stop
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
        "signal_type": signal_type,
    }
    if max_leverage is not None:
        metrics["max_leverage"] = float(max_leverage)
    if max_drawdown is not None:
        metrics["max_drawdown"] = float(max_drawdown)
    if signal_type == "arima":
        metrics["arima_order"] = tuple(int(x) for x in arima_order)

    # record additional parameters for newer signals
    if signal_type in {"vol_scaled_momentum", "regime_switch"}:
        metrics["vol_window"] = int(vol_window)
    if signal_type == "regime_switch":
        metrics["vol_threshold"] = float(vol_threshold)
    if signal_type == "ewma_momentum":
        metrics["ewma_span"] = int(ewma_span)
    if signal_type == "ma_crossover":
        metrics["ma_short"] = int(ma_short)
        metrics["ma_long"] = int(ma_long)
    return {"daily": daily, "equity": equity, "weights": weights, "metrics": metrics}
