"""Signal combination engine for blending multiple alpha signals.

Provides two combination modes:

1. **Equal-weight blend** -- z-score normalise each signal and average.
   No fitting required. Robust baseline for multi-signal strategies.

2. **Walk-forward optimised blend** -- rolling train/test split where
   signal weights are optimised on each training window using purged
   CV Sharpe, then applied out-of-sample on the test window. The
   out-of-sample returns are stitched together to produce a realistic
   performance estimate.

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 12.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .risk_model import RiskConfig, apply_risk_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinedResult:
    """Output of a combined signal backtest.

    Attributes
    ----------
    equity : pd.Series
        Cumulative return series.
    daily : pd.Series
        Daily return series.
    weights_used : dict[str, float]
        Signal weights applied (for equal-weight, all equal).
    metrics : dict
        Performance metrics from the backtest.
    mode : str
        ``"equal_weight"`` or ``"walk_forward"``.
    """

    equity: pd.Series
    daily: pd.Series
    weights_used: dict[str, float]
    metrics: dict[str, Any]
    mode: str


@dataclass
class WalkForwardResult:
    """Output of a walk-forward optimised backtest.

    Attributes
    ----------
    equity : pd.Series
        Stitched out-of-sample cumulative return series.
    daily : pd.Series
        Stitched out-of-sample daily returns.
    window_results : list[dict]
        Per-window details: train/test dates, chosen weights, OOS Sharpe.
    overall_metrics : dict
        Aggregate metrics on the stitched OOS returns.
    """

    equity: pd.Series
    daily: pd.Series
    window_results: list[dict[str, Any]] = field(default_factory=list)
    overall_metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Signal z-score helpers
# ---------------------------------------------------------------------------


def _zscore_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score normalisation per row.

    Subtracts the row mean and divides by the row standard deviation.
    Rows with zero variance are set to zero.
    """
    row_mean = signal.mean(axis=1)
    row_std = signal.std(axis=1).replace(0, np.nan)
    z = signal.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)
    return z


def compute_signals(
    prices: pd.DataFrame,
    signal_names: list[str],
    lookback: int = 20,
    ewma_span: int = 20,
    ma_short: int = 10,
    ma_long: int = 50,
    vol_window: int = 20,
    vol_threshold: float = 0.02,
    skip: int = 21,
) -> dict[str, pd.DataFrame]:
    """Compute multiple named signals and return as a dict.

    Each signal is z-score normalised cross-sectionally.

    Parameters
    ----------
    prices : DataFrame
        Price data (datetime index, asset columns).
    signal_names : list[str]
        Signal names matching the backtest signal registry.
    lookback, ewma_span, ma_short, ma_long, vol_window, vol_threshold, skip
        Signal-specific parameters.

    Returns
    -------
    dict[str, DataFrame]
        Mapping from signal name to z-scored signal DataFrame.
    """
    from . import signals as sig_module

    _dispatch = {
        "momentum": lambda p: sig_module.momentum_signal(p, lookback=lookback),
        "mean_reversion": lambda p: sig_module.mean_reversion_signal(p, lookback=lookback),
        "ewma_momentum": lambda p: sig_module.ewma_momentum_signal(p, span=ewma_span),
        "volatility": lambda p: sig_module.volatility_signal(p, lookback=lookback),
        "vol_scaled_momentum": lambda p: sig_module.volatility_scaled_momentum_signal(
            p, lookback=lookback, vol_window=vol_window
        ),
        "regime_switch": lambda p: sig_module.regime_switch_signal(
            p, lookback=lookback, vol_window=vol_window, vol_threshold=vol_threshold
        ),
        "ma_crossover": lambda p: sig_module.moving_average_crossover_signal(
            p, ma_short=ma_short, ma_long=ma_long
        ),
        "skip_month_momentum": lambda p: sig_module.skip_month_momentum_signal(
            p, lookback=lookback, skip=skip
        ),
        "residual_momentum": lambda p: sig_module.residual_momentum_signal(
            p, lookback=lookback, skip=skip
        ),
    }

    result = {}
    for name in signal_names:
        if name not in _dispatch:
            raise ValueError(f"Unknown signal: {name}. Choose from {list(_dispatch.keys())}")
        raw = _dispatch[name](prices)
        result[name] = _zscore_signal(raw)
        logger.debug("Computed signal '%s': shape %s", name, raw.shape)

    return result


# ---------------------------------------------------------------------------
# Equal-weight blend
# ---------------------------------------------------------------------------


def blend_equal_weight(
    prices: pd.DataFrame,
    signal_names: list[str],
    max_gross: float = 1.0,
    cost_bps: float = 10.0,
    rebalance: str = "daily",
    impact_model: str = "proportional",
    risk_config: RiskConfig | None = None,
    **signal_kwargs,
) -> CombinedResult:
    """Run a backtest using an equal-weight blend of multiple signals.

    Each signal is z-scored cross-sectionally, then the signals are
    averaged to produce a single combined score. The combined score
    is passed through the risk model pipeline for portfolio construction.

    Parameters
    ----------
    prices : DataFrame
        Price data.
    signal_names : list[str]
        Signals to blend.
    max_gross : float
        Gross exposure cap.
    cost_bps : float
        Transaction cost in basis points.
    rebalance : str
        Rebalance frequency.
    impact_model : str
        Cost model type.
    risk_config : RiskConfig or None
        Risk model configuration. Pass ``RiskConfig()`` for defaults
        or ``NO_RISK`` to disable all transforms.
    **signal_kwargs
        Passed to ``compute_signals``.

    Returns
    -------
    CombinedResult
    """
    if len(signal_names) < 2:
        raise ValueError("Need at least 2 signals to blend.")

    signals = compute_signals(prices, signal_names, **signal_kwargs)
    dfs = list(signals.values())
    combined: pd.DataFrame = dfs[0].copy()
    for df in dfs[1:]:
        combined = combined + df
    combined = combined / len(dfs)

    # Build weights via risk model or simple normalisation
    if risk_config is not None:
        weights = apply_risk_model(combined, prices, config=risk_config, max_gross=max_gross)
    else:
        row_abs_sum = combined.abs().sum(axis=1).replace(0, np.nan)
        weights = combined.div(row_abs_sum, axis=0).fillna(0.0) * max_gross

    # Run backtest with precomputed weights
    returns = prices.pct_change()
    port_ret = (weights.shift(1) * returns).sum(axis=1).dropna()

    # Transaction costs
    turnover = weights.diff().abs().sum(axis=1)
    if impact_model == "sqrt":
        cost = turnover.apply(np.sqrt) * (cost_bps / 10_000)
    else:
        cost = turnover * (cost_bps / 10_000)
    port_ret = port_ret - cost.reindex(port_ret.index, fill_value=0.0)

    equity = (1 + port_ret).cumprod()

    # Compute metrics
    n_days = len(port_ret)
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1 if n_days > 0 else 0.0
    years = n_days / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else float("nan")
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = (port_ret.mean() / port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 0.0
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")
    hit_rate = (port_ret > 0).mean()
    wins = port_ret[port_ret > 0].mean() if (port_ret > 0).any() else 0.0
    losses = abs(port_ret[port_ret < 0].mean()) if (port_ret < 0).any() else 1.0
    win_loss = wins / losses if losses > 0 else float("nan")
    avg_turnover = turnover.mean()

    weight_dict = {name: 1.0 / len(signal_names) for name in signal_names}

    metrics = {
        "days": n_days,
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": float(hit_rate),
        "win_loss_ratio": float(win_loss),
        "turnover_daily": float(avg_turnover),
        "max_gross": max_gross,
        "cost_bps": cost_bps,
        "signals": signal_names,
        "weights": weight_dict,
        "mode": "equal_weight",
        "rebalance": rebalance,
        "impact_model": impact_model,
        "risk_config": risk_config.to_dict() if risk_config is not None else None,
    }

    return CombinedResult(
        equity=equity,
        daily=port_ret,
        weights_used=weight_dict,
        metrics=metrics,
        mode="equal_weight",
    )


# ---------------------------------------------------------------------------
# Walk-forward optimised blend
# ---------------------------------------------------------------------------


def _generate_weight_grid(n_signals: int, step: float = 0.2) -> list[tuple[float, ...]]:
    """Generate all weight combinations that sum to 1.0.

    Each weight is a multiple of *step* in [0, 1].
    """
    levels = [round(i * step, 2) for i in range(int(1.0 / step) + 1)]
    grid = [
        combo
        for combo in itertools.product(levels, repeat=n_signals)
        if abs(sum(combo) - 1.0) < 1e-6
    ]
    return grid


def _backtest_with_signal_weights(
    prices: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    weights: dict[str, float],
    max_gross: float = 1.0,
    cost_bps: float = 10.0,
    impact_model: str = "proportional",
    risk_config: RiskConfig | None = None,
) -> pd.Series:
    """Run a quick backtest using pre-computed signals and weights.

    Returns the daily return series.
    """
    active = [(name, w) for name, w in weights.items() if w > 0]
    combined: pd.DataFrame = active[0][1] * signals[active[0][0]]
    for name, w in active[1:]:
        combined = combined + w * signals[name]

    if risk_config is not None:
        port_weights = apply_risk_model(combined, prices, config=risk_config, max_gross=max_gross)
    else:
        row_abs = combined.abs().sum(axis=1).replace(0, np.nan)
        port_weights = combined.div(row_abs, axis=0).fillna(0.0) * max_gross

    returns = prices.pct_change()
    port_ret = (port_weights.shift(1) * returns).sum(axis=1).dropna()

    turnover = port_weights.diff().abs().sum(axis=1)
    if impact_model == "sqrt":
        cost = turnover.apply(np.sqrt) * (cost_bps / 10_000)
    else:
        cost = turnover * (cost_bps / 10_000)

    port_ret = port_ret - cost.reindex(port_ret.index, fill_value=0.0)
    return port_ret


def walk_forward_optimize(
    prices: pd.DataFrame,
    signal_names: list[str],
    train_days: int = 504,
    test_days: int = 252,
    step_days: int = 252,
    weight_step: float = 0.2,
    max_gross: float = 1.0,
    cost_bps: float = 10.0,
    impact_model: str = "proportional",
    risk_config: RiskConfig | None = None,
    **signal_kwargs,
) -> WalkForwardResult:
    """Walk-forward optimisation of signal blend weights.

    Splits the price history into rolling train/test windows. On each
    training window, evaluates all weight combinations on the weight
    grid and selects the one with the highest Sharpe ratio. The winning
    weights are applied to the test window. Out-of-sample returns from
    all windows are stitched together.

    Parameters
    ----------
    prices : DataFrame
        Price data.
    signal_names : list[str]
        Signals to blend (at least 2).
    train_days : int
        Training window length in trading days (default 504, ~2 years).
    test_days : int
        Test window length in trading days (default 252, ~1 year).
    step_days : int
        Step size for rolling the window forward (default 252).
    weight_step : float
        Granularity of the weight grid (default 0.2).
    max_gross : float
        Gross exposure cap.
    cost_bps : float
        Transaction cost in basis points.
    impact_model : str
        Cost model type.
    **signal_kwargs
        Passed to ``compute_signals``.

    Returns
    -------
    WalkForwardResult
    """
    if len(signal_names) < 2:
        raise ValueError("Need at least 2 signals to blend.")

    n = len(prices)
    min_required = train_days + test_days
    if n < min_required:
        raise ValueError(f"Need at least {min_required} days, got {n}.")

    # Pre-compute all signals once
    signals = compute_signals(prices, signal_names, **signal_kwargs)
    weight_grid = _generate_weight_grid(len(signal_names), step=weight_step)

    logger.info(
        "Walk-forward: %d signals, %d weight combos, train=%d, test=%d, step=%d",
        len(signal_names),
        len(weight_grid),
        train_days,
        test_days,
        step_days,
    )

    oos_returns_list: list[pd.Series] = []
    window_results: list[dict] = []

    start = 0
    window_num = 0

    while start + train_days + test_days <= n:
        train_end = start + train_days
        test_end = min(train_end + test_days, n)

        train_prices = prices.iloc[start:train_end]
        test_prices = prices.iloc[train_end - 1 : test_end]  # overlap by 1 for pct_change

        train_signals = {name: sig.iloc[start:train_end] for name, sig in signals.items()}
        test_signals = {name: sig.iloc[train_end - 1 : test_end] for name, sig in signals.items()}

        # Find best weights on training data
        best_sharpe = -np.inf
        best_weights = None

        for combo in weight_grid:
            w = {name: combo[i] for i, name in enumerate(signal_names)}
            try:
                ret = _backtest_with_signal_weights(
                    train_prices,
                    train_signals,
                    w,
                    max_gross=max_gross,
                    cost_bps=cost_bps,
                    impact_model=impact_model,
                    risk_config=risk_config,
                )
                if len(ret) < 20 or ret.std() == 0:
                    continue
                sr = ret.mean() / ret.std() * np.sqrt(252)
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_weights = w
            except (ValueError, ZeroDivisionError, IndexError):
                continue

        if best_weights is None:
            best_weights = {name: 1.0 / len(signal_names) for name in signal_names}
            best_sharpe = float("nan")

        # Apply best weights to test window
        oos_ret = _backtest_with_signal_weights(
            test_prices,
            test_signals,
            best_weights,
            max_gross=max_gross,
            cost_bps=cost_bps,
            impact_model=impact_model,
            risk_config=risk_config,
        )
        oos_returns_list.append(oos_ret)

        train_start_date = str(prices.index[start].date())
        train_end_date = str(prices.index[train_end - 1].date())
        test_start_date = str(prices.index[train_end].date())
        test_end_date = str(prices.index[test_end - 1].date())

        window_info = {
            "window": window_num,
            "train_start": train_start_date,
            "train_end": train_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "best_weights": best_weights,
            "train_sharpe": float(best_sharpe),
            "oos_sharpe": float(
                oos_ret.mean() / oos_ret.std() * np.sqrt(252) if oos_ret.std() > 0 else 0.0
            ),
            "oos_days": len(oos_ret),
        }
        window_results.append(window_info)

        logger.info(
            "Window %d: train %s..%s, test %s..%s, weights=%s, train_SR=%.2f, oos_SR=%.2f",
            window_num,
            train_start_date,
            train_end_date,
            test_start_date,
            test_end_date,
            {k: round(v, 2) for k, v in best_weights.items()},
            best_sharpe,
            window_info["oos_sharpe"],
        )

        window_num += 1
        start += step_days

    # Stitch OOS returns
    if not oos_returns_list:
        raise ValueError("No walk-forward windows produced results.")

    oos_daily = pd.concat(oos_returns_list)
    # Remove duplicate indices (overlapping windows)
    oos_daily = oos_daily[~oos_daily.index.duplicated(keep="first")]
    oos_daily = oos_daily.sort_index()
    oos_equity = (1 + oos_daily).cumprod()

    # Overall metrics
    n_days = len(oos_daily)
    years = n_days / 252
    total_ret = oos_equity.iloc[-1] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else float("nan")
    ann_vol = oos_daily.std() * np.sqrt(252)
    sharpe = oos_daily.mean() / oos_daily.std() * np.sqrt(252) if oos_daily.std() > 0 else 0.0
    rolling_max = oos_equity.cummax()
    max_dd = (oos_equity / rolling_max - 1).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    overall_metrics = {
        "oos_days": n_days,
        "oos_cagr": float(cagr),
        "oos_ann_vol": float(ann_vol),
        "oos_sharpe": float(sharpe),
        "oos_max_dd": float(max_dd),
        "oos_calmar": float(calmar),
        "n_windows": len(window_results),
        "signals": signal_names,
        "mode": "walk_forward",
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "max_gross": max_gross,
        "cost_bps": cost_bps,
        "impact_model": impact_model,
    }

    return WalkForwardResult(
        equity=oos_equity,
        daily=oos_daily,
        window_results=window_results,
        overall_metrics=overall_metrics,
    )
