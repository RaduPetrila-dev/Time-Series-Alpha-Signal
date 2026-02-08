"""Core backtesting engine for cross-sectional time-series strategies.

This module provides a modular backtest pipeline with:

* A **signal registry** for extensible signal dispatch.
* **Rebalance frequency** control (daily, weekly, monthly).
* **Proportional and square-root market-impact** cost models.
* **Leverage constraints** and **drawdown stops** as post-processing steps.
* A frozen :class:`BacktestResult` dataclass for structured output.

All portfolio construction assumes a dollar-neutral long/short book whose
gross exposure is controlled by ``max_gross``.

.. note::
   Educational project — not investment advice.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import optimizer, signals

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Immutable container for backtest outputs.

    Attributes
    ----------
    daily : Series
        Daily net portfolio returns.
    equity : Series
        Cumulative equity curve (starts at 1.0).
    weights : DataFrame
        Portfolio weights for each asset on each day.
    metrics : dict[str, Any]
        Summary statistics (CAGR, Sharpe, max drawdown, turnover, etc.).
    stopped : bool
        ``True`` if the drawdown stop was triggered during the backtest.
    """

    daily: pd.Series
    equity: pd.Series
    weights: pd.DataFrame
    metrics: dict[str, Any]
    stopped: bool = False


# ---------------------------------------------------------------------------
# Rebalance frequency
# ---------------------------------------------------------------------------


class RebalanceFrequency(Enum):
    """Supported rebalance cadences."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


def _apply_rebalance_mask(
    weights: pd.DataFrame,
    frequency: RebalanceFrequency,
) -> pd.DataFrame:
    """Zero out weight changes on non-rebalance days.

    On days where no rebalance occurs the weights are carried forward from
    the most recent rebalance date.  This avoids unrealistic daily turnover
    for strategies that only trade weekly or monthly.

    Parameters
    ----------
    weights : DataFrame
        Raw daily weights produced by signal + normalisation.
    frequency : RebalanceFrequency
        How often the portfolio is allowed to trade.

    Returns
    -------
    DataFrame
        Weights with forward-fill on non-rebalance dates.
    """
    if frequency == RebalanceFrequency.DAILY:
        return weights

    idx = weights.index
    if frequency == RebalanceFrequency.WEEKLY:
        rebalance_mask = idx.to_series().dt.dayofweek == 0  # Monday
    elif frequency == RebalanceFrequency.MONTHLY:
        rebalance_mask = idx.to_series().dt.is_month_start
    else:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")

    # on non-rebalance days, carry forward the previous weights
    result = weights.copy()
    result.loc[~rebalance_mask.values] = np.nan
    return result.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------

# Maps a signal name to (callable, set_of_accepted_kwargs).  New signals
# are registered via :func:`register_signal`.
_SIGNAL_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


def register_signal(
    name: str,
    func: Callable[..., pd.DataFrame],
) -> None:
    """Register a signal function under *name*.

    Parameters
    ----------
    name : str
        Key used in ``signal_type`` to select this signal.
    func : callable
        Function with signature ``(prices: DataFrame, **kwargs) -> DataFrame``.
    """
    _SIGNAL_REGISTRY[name] = func


def _register_builtin_signals() -> None:
    """Populate the registry with signals shipped in :mod:`signals`."""
    register_signal("momentum", signals.momentum_signal)
    register_signal("mean_reversion", signals.mean_reversion_signal)
    register_signal("arima", signals.arima_signal)
    register_signal("volatility", signals.volatility_signal)
    register_signal(
        "vol_scaled_momentum", signals.volatility_scaled_momentum_signal
    )
    register_signal("regime_switch", signals.regime_switch_signal)
    register_signal("ewma_momentum", signals.ewma_momentum_signal)
    register_signal("ma_crossover", signals.moving_average_crossover_signal)


# Populate on import so the registry is ready when the module loads.
_register_builtin_signals()


def compute_signal(
    prices: pd.DataFrame,
    signal_type: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Dispatch to the registered signal function.

    Parameters
    ----------
    prices : DataFrame
        Asset price series indexed by datetime.
    signal_type : str
        Key in the signal registry.
    **kwargs
        Forwarded to the signal function.

    Returns
    -------
    DataFrame
        Cross-sectional signal scores.

    Raises
    ------
    ValueError
        If *signal_type* is not found in the registry.
    """
    if signal_type not in _SIGNAL_REGISTRY:
        available = ", ".join(sorted(_SIGNAL_REGISTRY))
        raise ValueError(
            f"Unknown signal_type '{signal_type}'. "
            f"Available signals: {available}"
        )
    logger.debug("Computing signal: %s with kwargs %s", signal_type, kwargs)
    return _SIGNAL_REGISTRY[signal_type](prices, **kwargs)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def normalize_weights(
    scores: pd.DataFrame,
    max_gross: float = 1.0,
    rank_method: str = "average",
) -> pd.DataFrame:
    """Convert cross-sectional scores to dollar-neutral portfolio weights.

    Scores are ranked cross-sectionally, centered to zero mean, and scaled
    so that the sum of absolute weights equals ``max_gross``.

    Parameters
    ----------
    scores : DataFrame
        Cross-sectional signal scores.
    max_gross : float, default 1.0
        Target gross exposure (sum of absolute weights).
    rank_method : str, default "average"
        Ranking method passed to :meth:`DataFrame.rank`.  Using ``"average"``
        treats ties symmetrically.  ``"first"`` introduces an arbitrary
        tiebreaker based on column order.

    Returns
    -------
    DataFrame
        Daily portfolio weights per asset.
    """
    ranks = scores.rank(axis=1, method=rank_method)
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    denom = centered.abs().sum(axis=1).replace(0.0, np.nan)
    raw = centered.div(denom, axis=0).fillna(0.0)
    return raw * max_gross


def construct_portfolio(
    scores: pd.DataFrame,
    max_gross: float = 1.0,
    max_leverage: float | None = None,
    rebalance: RebalanceFrequency = RebalanceFrequency.DAILY,
) -> pd.DataFrame:
    """Build portfolio weights from signal scores.

    Combines normalisation, leverage constraints, and rebalance frequency
    into a single call.

    Parameters
    ----------
    scores : DataFrame
        Cross-sectional signal scores.
    max_gross : float, default 1.0
        Gross exposure target.
    max_leverage : float, optional
        If provided, daily gross exposure is capped at this value.
    rebalance : RebalanceFrequency, default DAILY
        How often the portfolio is allowed to trade.

    Returns
    -------
    DataFrame
        Final portfolio weights.
    """
    weights = normalize_weights(scores, max_gross=max_gross)

    if max_leverage is not None:
        weights = optimizer.enforce_leverage(weights, max_leverage=max_leverage)

    weights = _apply_rebalance_mask(weights, rebalance)

    return weights


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


def _proportional_cost(
    turnover: pd.Series,
    bps: float,
) -> pd.Series:
    """Flat basis-point cost proportional to turnover.

    Parameters
    ----------
    turnover : Series
        Daily turnover (sum of absolute weight changes).
    bps : float
        Cost in basis points (1 bps = 0.01%).

    Returns
    -------
    Series
        Daily transaction cost.
    """
    return turnover * (bps / 1e4)


def _sqrt_impact_cost(
    turnover: pd.Series,
    bps: float,
) -> pd.Series:
    r"""Square-root market impact cost.

    Models the empirical observation that market impact scales with the
    square root of trade size:

    .. math::
        \text{cost}_t = \frac{\text{bps}}{10\,000} \sqrt{\text{turnover}_t}

    Parameters
    ----------
    turnover : Series
        Daily turnover.
    bps : float
        Impact coefficient in basis points.

    Returns
    -------
    Series
        Daily impact cost.
    """
    return (bps / 1e4) * np.sqrt(turnover)


# ---------------------------------------------------------------------------
# PnL simulation
# ---------------------------------------------------------------------------


def simulate(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
    impact_model: str = "proportional",
) -> pd.Series:
    """Compute daily portfolio returns net of transaction costs.

    Parameters
    ----------
    returns : DataFrame
        Daily asset returns aligned to *weights*.
    weights : DataFrame
        Portfolio weights.  Returns are computed using the previous day's
        weights (``weights.shift(1)``) to avoid lookahead.
    cost_bps : float, default 10.0
        Transaction cost parameter in basis points.
    impact_model : {"proportional", "sqrt"}, default "proportional"
        Cost model.  ``"proportional"`` charges a flat per-unit cost on
        turnover.  ``"sqrt"`` uses a square-root impact model.

    Returns
    -------
    Series
        Daily net portfolio returns.

    Raises
    ------
    ValueError
        If *impact_model* is not recognised.
    """
    gross = (weights.shift(1) * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)

    if impact_model == "proportional":
        cost = _proportional_cost(turnover, cost_bps)
    elif impact_model == "sqrt":
        cost = _sqrt_impact_cost(turnover, cost_bps)
    else:
        raise ValueError(
            f"Unknown impact_model '{impact_model}'. "
            "Choose 'proportional' or 'sqrt'."
        )

    net = gross - cost
    return net


# ---------------------------------------------------------------------------
# Drawdown stop
# ---------------------------------------------------------------------------


def apply_drawdown_stop(
    daily: pd.Series,
    max_drawdown: float,
) -> tuple[pd.Series, bool]:
    """Zero out returns after a drawdown breach.

    When the cumulative equity curve falls below
    ``-(max_drawdown)`` from its running peak, all subsequent returns
    are set to zero (the strategy goes flat).

    Parameters
    ----------
    daily : Series
        Daily portfolio returns.
    max_drawdown : float
        Positive fraction (e.g. 0.15 for -15%).

    Returns
    -------
    daily : Series
        Adjusted daily returns (zeroed after breach).
    stopped : bool
        ``True`` if the drawdown stop was triggered.
    """
    equity = (1 + daily).cumprod()
    dd = equity / equity.cummax() - 1.0
    breach_mask = dd < -abs(max_drawdown)

    if not breach_mask.any():
        return daily, False

    breach_start = breach_mask.idxmax()
    logger.info(
        "Drawdown stop triggered on %s (drawdown %.2f%%)",
        breach_start,
        float(dd.loc[breach_start]) * 100,
    )
    daily = daily.copy()
    daily.loc[breach_start:] = 0.0
    return daily, True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    daily: pd.Series,
    equity: pd.Series,
    weights: pd.DataFrame,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute summary performance statistics.

    Parameters
    ----------
    daily : Series
        Daily net returns.
    equity : Series
        Cumulative equity curve.
    weights : DataFrame
        Portfolio weights used during the backtest.
    extra : dict, optional
        Additional key/value pairs to include in the output.

    Returns
    -------
    dict
        Performance metrics including CAGR, annualised Sharpe ratio,
        maximum drawdown, annualised volatility, Calmar ratio, hit rate,
        average win/loss ratio, and mean daily turnover.
    """
    n_days = len(daily)

    # -- CAGR (guard against negative terminal equity) --
    final_equity = float(equity.iloc[-1]) if len(equity) > 0 else 1.0
    if final_equity > 0 and n_days > 0:
        cagr = float(final_equity ** (252 / n_days) - 1)
    else:
        cagr = -1.0

    # -- Annualised volatility --
    ann_vol = float(daily.std() * np.sqrt(252))

    # -- Sharpe ratio --
    vol = daily.std()
    sharpe = float(np.sqrt(252) * daily.mean() / vol) if vol > 0 else 0.0

    # -- Maximum drawdown --
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) > 0 else 0.0

    # -- Calmar ratio --
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # -- Hit rate --
    wins = (daily > 0).sum()
    losses = (daily < 0).sum()
    total_trades = wins + losses
    hit_rate = float(wins / total_trades) if total_trades > 0 else 0.0

    # -- Average win / average loss ratio --
    avg_win = float(daily[daily > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(daily[daily < 0].mean()) if losses > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # -- Turnover --
    turnover_daily = float(weights.diff().abs().sum(axis=1).mean())

    metrics: dict[str, Any] = {
        "days": n_days,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "win_loss_ratio": win_loss_ratio,
        "turnover_daily": turnover_daily,
    }

    if extra is not None:
        metrics.update(extra)

    return metrics


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort prices chronologically and warn about interior NaNs.

    Parameters
    ----------
    prices : DataFrame
        Raw price data.

    Returns
    -------
    DataFrame
        Sorted price data.

    Raises
    ------
    ValueError
        If *prices* is empty or has fewer than 2 rows.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError(
            "Price DataFrame must contain at least 2 rows; "
            f"got {len(prices)}."
        )

    prices = prices.sort_index()

    # Check for interior NaNs (after the first row which is expected NaN
    # from pct_change).
    interior_nans = prices.iloc[1:].isna().any().any()
    if interior_nans:
        warnings.warn(
            "Interior NaN values detected in price data. "
            "These will be forward-filled before computing returns.",
            stacklevel=2,
        )
        prices = prices.ffill()

    return prices


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def backtest(
    prices: pd.DataFrame,
    lookback: int = 20,
    max_gross: float = 1.0,
    cost_bps: float = 10.0,
    signal_type: str = "momentum",
    arima_order: Tuple[int, int, int] = (1, 0, 1),
    max_leverage: float | None = None,
    max_drawdown: float | None = None,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
    ewma_span: int = 20,
    ma_short: int = 10,
    ma_long: int = 50,
    rebalance: str = "daily",
    impact_model: str = "proportional",
) -> BacktestResult:
    """Run a time-series cross-sectional backtest.

    This is the main entry point that orchestrates the full pipeline:
    data validation, signal computation, portfolio construction, PnL
    simulation, optional drawdown stop, and metric computation.

    Parameters
    ----------
    prices : DataFrame
        Asset price series indexed by datetime.
    lookback : int, default 20
        Lookback period for simple signals (momentum, mean-reversion,
        volatility).
    max_gross : float, default 1.0
        Gross exposure limit when normalising cross-sectional ranks.
    cost_bps : float, default 10.0
        Transaction cost in basis points (1 bps = 0.01%).
    signal_type : str, default "momentum"
        Key in the signal registry.  Built-in options: ``"momentum"``,
        ``"mean_reversion"``, ``"arima"``, ``"volatility"``,
        ``"vol_scaled_momentum"``, ``"regime_switch"``,
        ``"ewma_momentum"``, ``"ma_crossover"``.
    arima_order : tuple of int, default (1, 0, 1)
        ARIMA ``(p, d, q)`` order for the ``"arima"`` signal.
    max_leverage : float, optional
        Cap on daily gross exposure.  ``None`` disables.
    max_drawdown : float, optional
        Drawdown stop as a positive fraction (e.g. 0.15 for -15%).
        ``None`` disables.
    vol_window : int, default 20
        Rolling window for volatility-based signals.
    vol_threshold : float, default 0.02
        Volatility threshold for ``"regime_switch"``.
    ewma_span : int, default 20
        EWMA span for ``"ewma_momentum"``.
    ma_short : int, default 10
        Short MA window for ``"ma_crossover"``.
    ma_long : int, default 50
        Long MA window for ``"ma_crossover"``.
    rebalance : {"daily", "weekly", "monthly"}, default "daily"
        Portfolio rebalance frequency.
    impact_model : {"proportional", "sqrt"}, default "proportional"
        Transaction cost model.

    Returns
    -------
    BacktestResult
        Frozen dataclass with ``daily``, ``equity``, ``weights``,
        ``metrics``, and ``stopped`` fields.
    """
    # -- validate --------------------------------------------------------
    prices = _validate_prices(prices)
    rebalance_freq = RebalanceFrequency(rebalance)

    # -- signal ----------------------------------------------------------
    signal_kwargs: dict[str, Any] = {}
    if signal_type in {"momentum", "mean_reversion", "volatility"}:
        signal_kwargs["lookback"] = lookback
    elif signal_type == "arima":
        signal_kwargs["order"] = arima_order
    elif signal_type == "vol_scaled_momentum":
        signal_kwargs["lookback"] = lookback
        signal_kwargs["vol_window"] = vol_window
    elif signal_type == "regime_switch":
        signal_kwargs["lookback"] = lookback
        signal_kwargs["vol_window"] = vol_window
        signal_kwargs["vol_threshold"] = vol_threshold
    elif signal_type == "ewma_momentum":
        signal_kwargs["span"] = ewma_span
    elif signal_type == "ma_crossover":
        signal_kwargs["ma_short"] = ma_short
        signal_kwargs["ma_long"] = ma_long

    scores = compute_signal(prices, signal_type, **signal_kwargs)

    # -- portfolio construction ------------------------------------------
    weights = construct_portfolio(
        scores,
        max_gross=max_gross,
        max_leverage=max_leverage,
        rebalance=rebalance_freq,
    )

    # -- PnL simulation --------------------------------------------------
    rets = prices.pct_change().fillna(0.0)
    daily = simulate(rets, weights, cost_bps=cost_bps, impact_model=impact_model)

    # -- drawdown stop ---------------------------------------------------
    stopped = False
    if max_drawdown is not None:
        daily, stopped = apply_drawdown_stop(daily, max_drawdown)
        if stopped:
            # zero out weights after the stop date for consistency
            equity_tmp = (1 + daily).cumprod()
            dd = equity_tmp / equity_tmp.cummax() - 1.0
            breach_mask = dd < -abs(max_drawdown)
            if breach_mask.any():
                breach_start = breach_mask.idxmax()
                weights.loc[breach_start:] = 0.0

    # -- equity curve ----------------------------------------------------
    equity = (1 + daily).cumprod()

    # -- metrics ---------------------------------------------------------
    extra: dict[str, Any] = {
        "max_gross": float(max_gross),
        "lookback": int(lookback),
        "cost_bps": float(cost_bps),
        "signal_type": signal_type,
        "rebalance": rebalance,
        "impact_model": impact_model,
    }
    if max_leverage is not None:
        extra["max_leverage"] = float(max_leverage)
    if max_drawdown is not None:
        extra["max_drawdown"] = float(max_drawdown)
    if signal_type == "arima":
        extra["arima_order"] = tuple(int(x) for x in arima_order)
    if signal_type in {"vol_scaled_momentum", "regime_switch"}:
        extra["vol_window"] = int(vol_window)
    if signal_type == "regime_switch":
        extra["vol_threshold"] = float(vol_threshold)
    if signal_type == "ewma_momentum":
        extra["ewma_span"] = int(ewma_span)
    if signal_type == "ma_crossover":
        extra["ma_short"] = int(ma_short)
        extra["ma_long"] = int(ma_long)

    metrics = compute_metrics(daily, equity, weights, extra=extra)

    return BacktestResult(
        daily=daily,
        equity=equity,
        weights=weights,
        metrics=metrics,
        stopped=stopped,
    )
