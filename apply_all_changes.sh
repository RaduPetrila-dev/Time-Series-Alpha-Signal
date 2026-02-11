#!/bin/bash
# ============================================================
# Run this in your test branch codespace.
# It applies ALL changes: risk_model.py, signals patch,
# combiner.py, run_combined.py, test_risk_model.py,
# __init__.py, workflow, and README.
# ============================================================
set -e

echo "=== Applying all changes to test branch ==="

# ----------------------------------------------------------------
# 1. NEW FILE: src/time_series_alpha_signal/risk_model.py
# ----------------------------------------------------------------
cat > src/time_series_alpha_signal/risk_model.py << 'PYEOF'
"""Risk model for portfolio construction.

Transforms raw signal scores into risk-adjusted portfolio weights.
Each transform is independent and composable via :func:`apply_risk_model`.

Transforms (applied in order):

1. **Signal clipping** -- cap extreme z-scores to reduce concentration.
2. **Signal smoothing** -- exponential decay to reduce turnover.
3. **Inverse-volatility weighting** -- scale positions by 1/vol for
   equal risk contribution.
4. **Volatility targeting** -- scale the portfolio to a target
   annualised volatility.

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
Pedersen, L. H. (2015). *Efficiently Inefficient*, Ch. 8 (Risk Management).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk model transforms.

    Parameters
    ----------
    clip_zscore : float
        Cap absolute signal values at this many standard deviations.
        Set to 0.0 to disable. Default 2.0.
    smooth_halflife : int
        Halflife in days for exponential smoothing of the signal.
        Set to 0 to disable. Default 5.
    inverse_vol : bool
        Scale positions by inverse realised volatility. Default True.
    vol_lookback : int
        Lookback window for volatility estimation. Default 20.
    vol_target : float
        Target annualised portfolio volatility. Set to 0.0 to disable.
        Default 0.15 (15%).
    vol_target_lookback : int
        Lookback for portfolio volatility estimation used in
        vol targeting. Default 63 (quarterly).
    vol_floor : float
        Minimum volatility estimate to avoid division by near-zero.
        Default 0.01 (1% annualised).
    max_leverage : float
        Hard cap on gross leverage after vol targeting. Default 2.0.
    """

    clip_zscore: float = 2.0
    smooth_halflife: int = 5
    inverse_vol: bool = True
    vol_lookback: int = 20
    vol_target: float = 0.15
    vol_target_lookback: int = 63
    vol_floor: float = 0.01
    max_leverage: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a dictionary."""
        return {
            "clip_zscore": self.clip_zscore,
            "smooth_halflife": self.smooth_halflife,
            "inverse_vol": self.inverse_vol,
            "vol_lookback": self.vol_lookback,
            "vol_target": self.vol_target,
            "vol_target_lookback": self.vol_target_lookback,
            "vol_floor": self.vol_floor,
            "max_leverage": self.max_leverage,
        }


def clip_signal(
    signal: pd.DataFrame,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Clip signal values to [-n_std, +n_std] cross-sectionally."""
    if n_std <= 0:
        return signal
    return signal.clip(lower=-n_std, upper=n_std)


def smooth_signal(
    signal: pd.DataFrame,
    halflife: int = 5,
) -> pd.DataFrame:
    """Apply exponential weighted moving average to reduce turnover."""
    if halflife <= 0:
        return signal
    return signal.ewm(halflife=halflife, min_periods=1).mean()


def inverse_vol_weight(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_floor: float = 0.01,
) -> pd.DataFrame:
    """Scale each position by inverse realised volatility."""
    returns = prices.pct_change()
    realised_vol = returns.rolling(lookback, min_periods=max(lookback // 2, 2)).std()
    ann_vol = realised_vol * np.sqrt(252)
    ann_vol = ann_vol.clip(lower=vol_floor)
    inv_vol = 1.0 / ann_vol
    weighted = signal * inv_vol
    weighted = weighted.reindex(signal.index).fillna(0.0)
    return weighted


def vol_target_scale(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    lookback: int = 63,
    vol_floor: float = 0.01,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale portfolio weights to achieve a target annualised volatility."""
    if target_vol <= 0:
        return weights
    returns = prices.pct_change()
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    realised_vol = port_ret.rolling(lookback, min_periods=max(lookback // 2, 2)).std()
    ann_port_vol = (realised_vol * np.sqrt(252)).clip(lower=vol_floor)
    scale = (target_vol / ann_port_vol).clip(upper=max_leverage)
    scaled = weights.mul(scale, axis=0)
    return scaled


def normalise_weights(
    weights: pd.DataFrame,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """Normalise portfolio weights to respect gross exposure constraint."""
    row_abs = weights.abs().sum(axis=1).replace(0, np.nan)
    normed = weights.div(row_abs, axis=0).fillna(0.0) * max_gross
    return normed


def apply_risk_model(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    config: RiskConfig | None = None,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """Apply the full risk model pipeline to raw signal scores.

    Pipeline order:
    1. Clip extreme signal values.
    2. Smooth the signal to reduce turnover.
    3. Scale by inverse volatility (equal risk contribution).
    4. Normalise to max_gross.
    5. Apply volatility targeting.
    """
    if config is None:
        config = RiskConfig()

    w = signal.copy()

    if config.clip_zscore > 0:
        w = clip_signal(w, n_std=config.clip_zscore)
        logger.debug("Clipped signal at +/- %.1f std", config.clip_zscore)

    if config.smooth_halflife > 0:
        w = smooth_signal(w, halflife=config.smooth_halflife)
        logger.debug("Smoothed signal with halflife=%d", config.smooth_halflife)

    if config.inverse_vol:
        w = inverse_vol_weight(
            w, prices, lookback=config.vol_lookback, vol_floor=config.vol_floor,
        )
        logger.debug("Applied inverse-vol weighting, lookback=%d", config.vol_lookback)

    w = normalise_weights(w, max_gross=max_gross)

    if config.vol_target > 0:
        w = vol_target_scale(
            w, prices, target_vol=config.vol_target,
            lookback=config.vol_target_lookback,
            vol_floor=config.vol_floor, max_leverage=config.max_leverage,
        )
        logger.debug(
            "Applied vol targeting: target=%.1f%%, max_leverage=%.1f",
            config.vol_target * 100, config.max_leverage,
        )

    return w
PYEOF
echo "Created risk_model.py"

# ----------------------------------------------------------------
# 2. PATCH: src/time_series_alpha_signal/signals.py
#    Add skip_month_momentum_signal and residual_momentum_signal
#    after the existing momentum_signal function.
# ----------------------------------------------------------------
python3 << 'PATCH_SIGNALS'
import re

with open("src/time_series_alpha_signal/signals.py", "r") as f:
    content = f.read()

# Check if already patched
if "skip_month_momentum_signal" in content:
    print("signals.py already patched, skipping.")
else:
    new_signals = '''

def skip_month_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Cross-sectional momentum skipping the most recent month.

    Computes the return from *lookback* days ago to *skip* days ago,
    excluding the most recent *skip* days to avoid short-term reversal
    contamination (Jegadeesh, 1990).

    The classic 12-1 configuration uses lookback=252, skip=21.

    Parameters
    ----------
    prices : DataFrame
        Price data (datetime index, asset columns).
    lookback : int
        Total formation period in trading days (default 252).
    skip : int
        Recent days to exclude (default 21, ~1 month).

    Returns
    -------
    DataFrame
        Cross-sectional signal scores (lagged by 1 day).
    """
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(skip, "skip")
    if skip >= lookback:
        raise ValueError(f"skip ({skip}) must be < lookback ({lookback}).")
    formation_ret = prices.shift(skip) / prices.shift(lookback) - 1
    return formation_ret.shift(1)


def residual_momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Cross-sectional residual momentum (idiosyncratic momentum).

    For each day, regresses each asset's trailing returns against the
    equal-weight market return to obtain residuals. Then computes
    momentum on the residual returns, skipping the most recent *skip*
    days to avoid short-term reversal.

    This captures stock-specific trends independent of broad market
    direction, reducing drawdowns during market-wide selloffs.

    Parameters
    ----------
    prices : DataFrame
        Price data (datetime index, asset columns).
    lookback : int
        Formation period in trading days (default 252).
    skip : int
        Recent days to exclude (default 21).

    Returns
    -------
    DataFrame
        Cross-sectional residual momentum scores (lagged by 1 day).

    References
    ----------
    Blitz, D., Huij, J. & Martens, M. (2011). Residual Momentum.
    *Journal of Empirical Finance*, 18(3), 506-521.
    """
    _validate_prices(prices)
    _validate_lookback(lookback)
    _validate_lookback(skip, "skip")
    if skip >= lookback:
        raise ValueError(f"skip ({skip}) must be < lookback ({lookback}).")

    rets = prices.pct_change()
    market_ret = rets.mean(axis=1)

    residuals = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    rolling_var = market_ret.rolling(lookback, min_periods=lookback // 2).var()

    for col in rets.columns:
        rolling_cov = rets[col].rolling(lookback, min_periods=lookback // 2).cov(market_ret)
        beta = rolling_cov / rolling_var.clip(lower=1e-10)
        residuals[col] = rets[col] - beta * market_ret

    residual_cum = residuals.rolling(lookback, min_periods=lookback // 2).sum()
    residual_cum_skip = residuals.rolling(skip, min_periods=1).sum()
    signal = residual_cum - residual_cum_skip

    return signal.shift(1)
'''

    # Insert after mean_reversion_signal (which comes after momentum_signal)
    # Find the end of mean_reversion_signal function
    pattern = r'(def mean_reversion_signal\(.*?\n    return rev\.shift\(1\))'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        insert_pos = match.end()
        content = content[:insert_pos] + new_signals + content[insert_pos:]
    else:
        # Fallback: insert before arima_signal
        pattern2 = r'\ndef arima_signal\('
        match2 = re.search(pattern2, content)
        if match2:
            content = content[:match2.start()] + new_signals + content[match2.start():]
        else:
            # Last resort: append
            content += new_signals

    with open("src/time_series_alpha_signal/signals.py", "w") as f:
        f.write(content)
    print("Patched signals.py with skip_month_momentum and residual_momentum")
PATCH_SIGNALS

# ----------------------------------------------------------------
# 3. REPLACE: src/time_series_alpha_signal/combiner.py
# ----------------------------------------------------------------
cp src/time_series_alpha_signal/combiner.py src/time_series_alpha_signal/combiner.py.bak

python3 << 'PATCH_COMBINER'
with open("src/time_series_alpha_signal/combiner.py", "r") as f:
    content = f.read()

# Add risk_model import if not present
if "from .risk_model import" not in content:
    content = content.replace(
        "import numpy as np\nimport pandas as pd\n\nlogger",
        "import numpy as np\nimport pandas as pd\n\nfrom .risk_model import RiskConfig, apply_risk_model\n\nlogger"
    )
    print("Added risk_model import")

# Add skip parameter to compute_signals
if "skip: int = 21," not in content:
    content = content.replace(
        "    vol_threshold: float = 0.02,\n) -> dict[str, pd.DataFrame]:",
        "    vol_threshold: float = 0.02,\n    skip: int = 21,\n) -> dict[str, pd.DataFrame]:"
    )
    # Update docstring
    content = content.replace(
        "    lookback, ewma_span, ma_short, ma_long, vol_window, vol_threshold\n        Signal-specific parameters.",
        "    lookback, ewma_span, ma_short, ma_long, vol_window, vol_threshold, skip\n        Signal-specific parameters."
    )
    print("Added skip parameter to compute_signals")

# Add new signals to dispatch
if "skip_month_momentum" not in content:
    content = content.replace(
        '        "ma_crossover": lambda p: sig_module.moving_average_crossover_signal(\n            p, ma_short=ma_short, ma_long=ma_long\n        ),\n    }',
        '        "ma_crossover": lambda p: sig_module.moving_average_crossover_signal(\n            p, ma_short=ma_short, ma_long=ma_long\n        ),\n        "skip_month_momentum": lambda p: sig_module.skip_month_momentum_signal(\n            p, lookback=lookback, skip=skip\n        ),\n        "residual_momentum": lambda p: sig_module.residual_momentum_signal(\n            p, lookback=lookback, skip=skip\n        ),\n    }'
    )
    print("Added skip_month_momentum and residual_momentum to dispatch")

# Add risk_config parameter to blend_equal_weight
if "risk_config: RiskConfig | None = None," not in content:
    content = content.replace(
        '    impact_model: str = "proportional",\n    **signal_kwargs,\n) -> CombinedResult:\n    """Run a backtest using an equal-weight blend',
        '    impact_model: str = "proportional",\n    risk_config: RiskConfig | None = None,\n    **signal_kwargs,\n) -> CombinedResult:\n    """Run a backtest using an equal-weight blend'
    )
    # Replace portfolio construction with risk model
    content = content.replace(
        "    # Build weights from combined signal (same logic as backtest engine)\n    row_abs_sum = combined.abs().sum(axis=1).replace(0, np.nan)\n    weights = combined.div(row_abs_sum, axis=0).fillna(0.0) * max_gross",
        "    # Build weights via risk model or simple normalisation\n    if risk_config is not None:\n        weights = apply_risk_model(\n            combined, prices, config=risk_config, max_gross=max_gross\n        )\n    else:\n        row_abs_sum = combined.abs().sum(axis=1).replace(0, np.nan)\n        weights = combined.div(row_abs_sum, axis=0).fillna(0.0) * max_gross"
    )
    # Add risk_config to metrics dict
    content = content.replace(
        '        "rebalance": rebalance,\n        "impact_model": impact_model,\n    }',
        '        "rebalance": rebalance,\n        "impact_model": impact_model,\n        "risk_config": risk_config.to_dict() if risk_config is not None else None,\n    }'
    )
    print("Added risk_config to blend_equal_weight")

# Add risk_config to _backtest_with_signal_weights
if 'risk_config: RiskConfig | None = None,\n) -> pd.Series:' not in content:
    content = content.replace(
        '    impact_model: str = "proportional",\n) -> pd.Series:\n    """Run a quick backtest using pre-computed signals and weights.\n\n    Returns the daily return series.\n    """\n    active = [(name, w) for name, w in weights.items() if w > 0]\n    combined: pd.DataFrame = active[0][1] * signals[active[0][0]]\n    for name, w in active[1:]:\n        combined = combined + w * signals[name]\n    row_abs = combined.abs().sum(axis=1).replace(0, np.nan)\n    port_weights = combined.div(row_abs, axis=0).fillna(0.0) * max_gross',
        '    impact_model: str = "proportional",\n    risk_config: RiskConfig | None = None,\n) -> pd.Series:\n    """Run a quick backtest using pre-computed signals and weights.\n\n    Returns the daily return series.\n    """\n    active = [(name, w) for name, w in weights.items() if w > 0]\n    combined: pd.DataFrame = active[0][1] * signals[active[0][0]]\n    for name, w in active[1:]:\n        combined = combined + w * signals[name]\n\n    if risk_config is not None:\n        port_weights = apply_risk_model(\n            combined, prices, config=risk_config, max_gross=max_gross\n        )\n    else:\n        row_abs = combined.abs().sum(axis=1).replace(0, np.nan)\n        port_weights = combined.div(row_abs, axis=0).fillna(0.0) * max_gross'
    )
    print("Added risk_config to _backtest_with_signal_weights")

# Add risk_config to walk_forward_optimize signature
if "risk_config: RiskConfig | None = None,\n    **signal_kwargs,\n) -> WalkForwardResult:" not in content:
    content = content.replace(
        '    impact_model: str = "proportional",\n    **signal_kwargs,\n) -> WalkForwardResult:',
        '    impact_model: str = "proportional",\n    risk_config: RiskConfig | None = None,\n    **signal_kwargs,\n) -> WalkForwardResult:'
    )
    print("Added risk_config to walk_forward_optimize")

# Pass risk_config to _backtest_with_signal_weights calls
content = content.replace(
    "                ret = _backtest_with_signal_weights(\n                    train_prices, train_signals, w,\n                    max_gross=max_gross, cost_bps=cost_bps,\n                    impact_model=impact_model,\n                )",
    "                ret = _backtest_with_signal_weights(\n                    train_prices, train_signals, w,\n                    max_gross=max_gross, cost_bps=cost_bps,\n                    impact_model=impact_model,\n                    risk_config=risk_config,\n                )"
)
content = content.replace(
    "        oos_ret = _backtest_with_signal_weights(\n            test_prices, test_signals, best_weights,\n            max_gross=max_gross, cost_bps=cost_bps,\n            impact_model=impact_model,\n        )",
    "        oos_ret = _backtest_with_signal_weights(\n            test_prices, test_signals, best_weights,\n            max_gross=max_gross, cost_bps=cost_bps,\n            impact_model=impact_model,\n            risk_config=risk_config,\n        )"
)

with open("src/time_series_alpha_signal/combiner.py", "w") as f:
    f.write(content)
print("Patched combiner.py")
PATCH_COMBINER

# ----------------------------------------------------------------
# 4. REPLACE: scripts/run_combined.py
# ----------------------------------------------------------------
cat > scripts/run_combined.py << 'PYEOF'
"""Run combined signal backtests on equity or crypto baskets.

Usage examples::

    # Equal-weight blend on FAANG
    python scripts/run_combined.py \
        --tickers AAPL,AMZN,GOOGL,META,NFLX \
        --start 2018-01-01 --end 2023-12-31 \
        --signals momentum,mean_reversion,ewma_momentum \
        --mode equal_weight \
        --output results/faang_combined_eqw

    # Walk-forward optimised on crypto
    python scripts/run_combined.py \
        --tickers BTC-USD,ETH-USD \
        --start 2018-01-01 --end 2023-12-31 \
        --signals momentum,mean_reversion,ewma_momentum,volatility \
        --mode walk_forward \
        --train-days 504 --test-days 252 \
        --output results/crypto_combined_wf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from time_series_alpha_signal.combiner import (  # noqa: E402
    blend_equal_weight,
    walk_forward_optimize,
)
from time_series_alpha_signal.data import load_csv_prices, load_yfinance_prices  # noqa: E402
from time_series_alpha_signal.risk_model import RiskConfig  # noqa: E402

logger = logging.getLogger(__name__)


def _save_plots(
    equity: pd.Series,
    daily: pd.Series,
    out_dir: Path,
    title_prefix: str = "",
) -> None:
    """Save equity curve, drawdown, and rolling Sharpe plots."""
    prefix = f"{title_prefix} " if title_prefix else ""

    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax, title=f"{prefix}Equity Curve")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("")
    fig.savefig(out_dir / "equity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    dd = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    dd.plot(ax=ax, title=f"{prefix}Drawdown", color="crimson")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.fill_between(dd.index, dd.values, alpha=0.3, color="crimson")
    fig.savefig(out_dir / "drawdown.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    rolling_sr = daily.rolling(63).mean() / daily.rolling(63).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_sr.plot(ax=ax, title=f"{prefix}Rolling 63-day Sharpe Ratio")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("")
    fig.savefig(out_dir / "rolling_sharpe.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _save_walk_forward_chart(
    window_results: list[dict],
    out_dir: Path,
) -> None:
    """Plot train vs OOS Sharpe per walk-forward window."""
    if not window_results:
        return
    windows = [w["window"] for w in window_results]
    train_sr = [w["train_sharpe"] for w in window_results]
    oos_sr = [w["oos_sharpe"] for w in window_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(windows))
    width = 0.35
    ax.bar(x - width / 2, train_sr, width, label="Train Sharpe", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, oos_sr, width, label="OOS Sharpe", color="coral", alpha=0.8)
    ax.set_xlabel("Window")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward: Train vs Out-of-Sample Sharpe")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w}" for w in windows])
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.legend()
    fig.savefig(out_dir / "walk_forward_sharpe.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run combined signal backtests."
    )
    parser.add_argument(
        "--tickers", type=str, required=True,
        help="Comma-separated tickers (e.g. AAPL,AMZN or BTC-USD,ETH-USD).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file (overrides --tickers).",
    )
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument(
        "--signals", type=str, required=True,
        help="Comma-separated signal names.",
    )
    parser.add_argument(
        "--mode", type=str, default="equal_weight",
        choices=["equal_weight", "walk_forward", "both"],
    )
    parser.add_argument("--max-gross", dest="max_gross", type=float, default=1.0)
    parser.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0)
    parser.add_argument("--impact-model", dest="impact_model", type=str, default="proportional")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ewma-span", dest="ewma_span", type=int, default=20)
    parser.add_argument("--skip", type=int, default=21, help="Skip days for skip_month_momentum.")
    parser.add_argument("--train-days", dest="train_days", type=int, default=504)
    parser.add_argument("--test-days", dest="test_days", type=int, default=252)
    parser.add_argument("--step-days", dest="step_days", type=int, default=252)
    parser.add_argument("--weight-step", dest="weight_step", type=float, default=0.2)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument(
        "--risk-model", dest="risk_model", action="store_true", default=False,
        help="Enable the risk model pipeline.",
    )
    parser.add_argument("--clip-zscore", dest="clip_zscore", type=float, default=2.0)
    parser.add_argument("--smooth-halflife", dest="smooth_halflife", type=int, default=5)
    parser.add_argument(
        "--no-inverse-vol", dest="inverse_vol", action="store_false", default=True,
    )
    parser.add_argument("--vol-lookback", dest="vol_lookback", type=int, default=20)
    parser.add_argument("--vol-target", dest="vol_target", type=float, default=0.15)
    parser.add_argument("--vol-target-lookback", dest="vol_target_lookback", type=int, default=63)
    parser.add_argument("--max-leverage", dest="max_leverage", type=float, default=2.0)

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=level,
        stream=sys.stderr,
    )

    risk_config = None
    if args.risk_model:
        risk_config = RiskConfig(
            clip_zscore=args.clip_zscore,
            smooth_halflife=args.smooth_halflife,
            inverse_vol=args.inverse_vol,
            vol_lookback=args.vol_lookback,
            vol_target=args.vol_target,
            vol_target_lookback=args.vol_target_lookback,
            max_leverage=args.max_leverage,
        )
        logger.info("Risk model enabled: %s", risk_config.to_dict())

    if args.csv:
        prices = load_csv_prices(args.csv)
    else:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        prices = load_yfinance_prices(tickers, start=args.start, end=args.end)

    signal_names = [s.strip() for s in args.signals.split(",") if s.strip()]
    out_dir = Path(args.output).resolve()

    signal_kwargs = {
        "lookback": args.lookback,
        "ewma_span": args.ewma_span,
        "skip": args.skip,
    }

    all_results = {}

    if args.mode in ("equal_weight", "both"):
        logger.info("Running equal-weight blend: %s", signal_names)
        eqw_dir = out_dir / "equal_weight" if args.mode == "both" else out_dir
        eqw_dir.mkdir(parents=True, exist_ok=True)

        eqw = blend_equal_weight(
            prices=prices,
            signal_names=signal_names,
            max_gross=args.max_gross,
            cost_bps=args.cost_bps,
            impact_model=args.impact_model,
            risk_config=risk_config,
            **signal_kwargs,
        )

        (eqw_dir / "metrics.json").write_text(json.dumps(eqw.metrics, indent=2))
        eqw.daily.to_csv(eqw_dir / "daily_returns.csv", header=True)
        _save_plots(eqw.equity, eqw.daily, eqw_dir, title_prefix="Equal-Weight")

        all_results["equal_weight"] = eqw.metrics
        logger.info(
            "Equal-weight Sharpe: %.3f, CAGR: %.4f",
            eqw.metrics["sharpe"],
            eqw.metrics["cagr"],
        )

    if args.mode in ("walk_forward", "both"):
        logger.info("Running walk-forward optimisation: %s", signal_names)
        wf_dir = out_dir / "walk_forward" if args.mode == "both" else out_dir
        wf_dir.mkdir(parents=True, exist_ok=True)

        wf = walk_forward_optimize(
            prices=prices,
            signal_names=signal_names,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            weight_step=args.weight_step,
            max_gross=args.max_gross,
            cost_bps=args.cost_bps,
            impact_model=args.impact_model,
            risk_config=risk_config,
            **signal_kwargs,
        )

        output = {
            "overall_metrics": wf.overall_metrics,
            "window_results": wf.window_results,
        }
        (wf_dir / "metrics.json").write_text(json.dumps(output, indent=2))
        wf.daily.to_csv(wf_dir / "daily_returns.csv", header=True)
        _save_plots(wf.equity, wf.daily, wf_dir, title_prefix="Walk-Forward")
        _save_walk_forward_chart(wf.window_results, wf_dir)

        all_results["walk_forward"] = wf.overall_metrics
        logger.info(
            "Walk-forward OOS Sharpe: %.3f, OOS CAGR: %.4f",
            wf.overall_metrics["oos_sharpe"],
            wf.overall_metrics["oos_cagr"],
        )

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
PYEOF
echo "Created run_combined.py"

# ----------------------------------------------------------------
# 5. PATCH: src/time_series_alpha_signal/__init__.py
#    Add risk_model exports
# ----------------------------------------------------------------
python3 << 'PATCH_INIT'
with open("src/time_series_alpha_signal/__init__.py", "r") as f:
    content = f.read()

if "risk_model" not in content:
    content = content.replace(
        "from .optimizer import enforce_leverage, optimize_ewma  # noqa: F401",
        "from .optimizer import enforce_leverage, optimize_ewma  # noqa: F401\nfrom .risk_model import RiskConfig, apply_risk_model  # noqa: F401"
    )
    if '"PurgedKFold",' in content and '"RiskConfig",' not in content:
        content = content.replace(
            '"PurgedKFold",',
            '"PurgedKFold",\n    "RiskConfig",\n    "apply_risk_model",'
        )
    with open("src/time_series_alpha_signal/__init__.py", "w") as f:
        f.write(content)
    print("Patched __init__.py")
else:
    print("__init__.py already has risk_model, skipping.")
PATCH_INIT

# ----------------------------------------------------------------
# 6. NEW FILE: tests/test_risk_model.py
# ----------------------------------------------------------------
cat > tests/test_risk_model.py << 'PYEOF'
"""Tests for the risk model module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_series_alpha_signal.risk_model import (
    RiskConfig,
    apply_risk_model,
    clip_signal,
    inverse_vol_weight,
    normalise_weights,
    smooth_signal,
    vol_target_scale,
)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", periods=500)
    data = 100 + np.cumsum(rng.standard_normal((500, 5)), axis=0)
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(5)])


@pytest.fixture
def synthetic_signal(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    data = rng.standard_normal(synthetic_prices.shape)
    return pd.DataFrame(
        data, index=synthetic_prices.index, columns=synthetic_prices.columns,
    )


class TestClipSignal:
    def test_values_within_bounds(self, synthetic_signal: pd.DataFrame) -> None:
        clipped = clip_signal(synthetic_signal, n_std=2.0)
        assert clipped.max().max() <= 2.0 + 1e-10
        assert clipped.min().min() >= -2.0 - 1e-10

    def test_disabled_when_zero(self, synthetic_signal: pd.DataFrame) -> None:
        result = clip_signal(synthetic_signal, n_std=0.0)
        pd.testing.assert_frame_equal(result, synthetic_signal)


class TestSmoothSignal:
    def test_reduces_daily_change(self, synthetic_signal: pd.DataFrame) -> None:
        smoothed = smooth_signal(synthetic_signal, halflife=5)
        raw_turnover = synthetic_signal.diff().abs().sum().sum()
        smooth_turnover = smoothed.diff().abs().sum().sum()
        assert smooth_turnover < raw_turnover

    def test_disabled_when_zero(self, synthetic_signal: pd.DataFrame) -> None:
        result = smooth_signal(synthetic_signal, halflife=0)
        pd.testing.assert_frame_equal(result, synthetic_signal)


class TestInverseVolWeight:
    def test_high_vol_gets_lower_weight(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        signal = pd.DataFrame(
            1.0, index=synthetic_prices.index, columns=synthetic_prices.columns,
        )
        weighted = inverse_vol_weight(signal, synthetic_prices, lookback=20)
        vols = (
            synthetic_prices.pct_change()
            .rolling(20, min_periods=10)
            .std()
            .iloc[-1]
        )
        highest_vol_asset = vols.idxmax()
        lowest_vol_asset = vols.idxmin()
        last_row = weighted.iloc[-1]
        assert last_row[lowest_vol_asset] > last_row[highest_vol_asset]


class TestVolTargetScale:
    def test_disabled_when_zero(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = pd.DataFrame(
            0.2, index=synthetic_prices.index, columns=synthetic_prices.columns,
        )
        result = vol_target_scale(weights, synthetic_prices, target_vol=0.0)
        pd.testing.assert_frame_equal(result, weights)


class TestNormaliseWeights:
    def test_gross_exposure(self) -> None:
        w = pd.DataFrame({"A": [1.0, -2.0], "B": [3.0, 0.5]})
        normed = normalise_weights(w, max_gross=1.0)
        gross = normed.abs().sum(axis=1)
        np.testing.assert_allclose(gross.values, [1.0, 1.0], atol=1e-10)


class TestApplyRiskModel:
    def test_default_config(
        self, synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = apply_risk_model(
            synthetic_signal, synthetic_prices, config=RiskConfig(),
        )
        assert weights.shape == synthetic_signal.shape
        assert np.isfinite(weights.values[100:]).all()

    def test_no_transforms(
        self, synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        config = RiskConfig(
            clip_zscore=0.0, smooth_halflife=0,
            inverse_vol=False, vol_target=0.0,
        )
        weights = apply_risk_model(
            synthetic_signal, synthetic_prices,
            config=config, max_gross=1.0,
        )
        row_abs = synthetic_signal.abs().sum(axis=1).replace(0, np.nan)
        expected = synthetic_signal.div(row_abs, axis=0).fillna(0.0)
        pd.testing.assert_frame_equal(weights, expected, atol=1e-10)


class TestRiskConfig:
    def test_to_dict_keys(self) -> None:
        config = RiskConfig()
        d = config.to_dict()
        expected_keys = {
            "clip_zscore", "smooth_halflife", "inverse_vol",
            "vol_lookback", "vol_target", "vol_target_lookback",
            "vol_floor", "max_leverage",
        }
        assert set(d.keys()) == expected_keys

    def test_frozen(self) -> None:
        config = RiskConfig()
        with pytest.raises(AttributeError):
            config.clip_zscore = 5.0  # type: ignore[misc]
PYEOF
echo "Created test_risk_model.py"

# ----------------------------------------------------------------
# 7. REPLACE: .github/workflows/combined_backtests.yml
# ----------------------------------------------------------------
mkdir -p .github/workflows
cat > .github/workflows/combined_backtests.yml << 'YMLEOF'
name: Combined Signal Backtests

on:
  workflow_dispatch:

permissions:
  contents: write

env:
  BROAD_TICKERS: >-
    AAPL,MSFT,GOOGL,META,NVDA,ADBE,CRM,INTC,JNJ,UNH,PFE,ABBV,MRK,TMO,JPM,BAC,GS,MS,BLK,AMZN,WMT,KO,PEP,MCD,NKE,CAT,HON,UPS,BA,XOM,CVX,COP,NEE,DUK,LIN,APD,AMT,PLD,NFLX,DIS

jobs:
  run-combined:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install
        run: |
          set -e
          python -m pip install --upgrade pip
          pip install -e ".[dev,arima]"
          pip install --upgrade yfinance

      - name: Broad 12-1 momentum baseline
        run: |
          python scripts/run_combined.py \
            --tickers ${{ env.BROAD_TICKERS }} \
            --start 2018-01-01 --end 2023-12-31 \
            --signals skip_month_momentum \
            --mode equal_weight \
            --lookback 252 --skip 21 \
            --cost-bps 10 --max-gross 1.0 \
            --output results/broad40_12_1_baseline \
            -v 2>&1

      - name: Broad 12-1 momentum with risk model
        run: |
          python scripts/run_combined.py \
            --tickers ${{ env.BROAD_TICKERS }} \
            --start 2018-01-01 --end 2023-12-31 \
            --signals skip_month_momentum \
            --mode equal_weight \
            --lookback 252 --skip 21 \
            --cost-bps 10 --max-gross 1.0 \
            --risk-model \
            --clip-zscore 2.0 --smooth-halflife 10 \
            --vol-target 0.10 --max-leverage 2.0 \
            --output results/broad40_12_1_risk \
            -v 2>&1

      - name: Broad residual momentum baseline
        run: |
          python scripts/run_combined.py \
            --tickers ${{ env.BROAD_TICKERS }} \
            --start 2018-01-01 --end 2023-12-31 \
            --signals residual_momentum \
            --mode equal_weight \
            --lookback 252 --skip 21 \
            --cost-bps 10 --max-gross 1.0 \
            --output results/broad40_resid_mom_baseline \
            -v 2>&1

      - name: Broad residual momentum with risk model
        run: |
          python scripts/run_combined.py \
            --tickers ${{ env.BROAD_TICKERS }} \
            --start 2018-01-01 --end 2023-12-31 \
            --signals residual_momentum \
            --mode equal_weight \
            --lookback 252 --skip 21 \
            --cost-bps 10 --max-gross 1.0 \
            --risk-model \
            --clip-zscore 2.0 --smooth-halflife 10 \
            --vol-target 0.10 --max-leverage 2.0 \
            --output results/broad40_resid_mom_risk \
            -v 2>&1

      - name: Broad 12-1 + residual + vol combo with risk model
        run: |
          python scripts/run_combined.py \
            --tickers ${{ env.BROAD_TICKERS }} \
            --start 2018-01-01 --end 2023-12-31 \
            --signals skip_month_momentum,residual_momentum,volatility \
            --mode equal_weight \
            --lookback 252 --skip 21 \
            --cost-bps 10 --max-gross 1.0 \
            --risk-model \
            --clip-zscore 2.0 --smooth-halflife 10 \
            --vol-target 0.10 --max-leverage 2.0 \
            --output results/broad40_combo_risk \
            -v 2>&1

      - name: Broad walk-forward 12-1 + residual + vol with risk model
        run: |
          python scripts/run_combined.py \
            --tickers ${{ env.BROAD_TICKERS }} \
            --start 2018-01-01 --end 2023-12-31 \
            --signals skip_month_momentum,residual_momentum,volatility \
            --mode walk_forward \
            --train-days 504 --test-days 252 --step-days 252 \
            --weight-step 0.5 \
            --lookback 252 --skip 21 \
            --cost-bps 10 --max-gross 1.0 \
            --risk-model \
            --clip-zscore 2.0 --smooth-halflife 10 \
            --vol-target 0.10 --max-leverage 2.0 \
            --output results/broad40_wf_combo_risk \
            -v 2>&1

      - name: Broad 20d momentum with risk model
        run: |
          python scripts/run_combined.py \
            --tickers ${{ env.BROAD_TICKERS }} \
            --start 2018-01-01 --end 2023-12-31 \
            --signals momentum,ewma_momentum \
            --mode equal_weight \
            --lookback 20 --ewma-span 30 \
            --cost-bps 10 --max-gross 1.0 \
            --risk-model \
            --clip-zscore 2.0 --smooth-halflife 10 \
            --vol-target 0.10 --max-leverage 2.0 \
            --output results/broad40_20d_mom_risk \
            -v 2>&1

      - name: FAANG equal-weight baseline
        run: |
          python scripts/run_combined.py \
            --tickers AAPL,AMZN,GOOGL,META,NFLX \
            --start 2018-01-01 --end 2023-12-31 \
            --signals momentum,mean_reversion,ewma_momentum \
            --mode equal_weight \
            --lookback 20 --ewma-span 30 \
            --cost-bps 10 --max-gross 1.0 \
            --output results/faang_combined_eqw \
            -v 2>&1

      - name: FAANG walk-forward with risk model
        run: |
          python scripts/run_combined.py \
            --tickers AAPL,AMZN,GOOGL,META,NFLX \
            --start 2018-01-01 --end 2023-12-31 \
            --signals momentum,mean_reversion,ewma_momentum,volatility \
            --mode walk_forward \
            --train-days 504 --test-days 252 --step-days 252 \
            --weight-step 0.25 \
            --lookback 20 --ewma-span 30 \
            --cost-bps 10 --max-gross 1.0 \
            --risk-model \
            --clip-zscore 2.0 --smooth-halflife 5 \
            --vol-target 0.10 --max-leverage 2.0 \
            --output results/faang_combined_wf_risk \
            -v 2>&1

      - name: Crypto equal-weight baseline
        run: |
          python scripts/run_combined.py \
            --tickers BTC-USD,ETH-USD \
            --start 2018-01-01 --end 2023-12-31 \
            --signals momentum,mean_reversion,ewma_momentum \
            --mode equal_weight \
            --lookback 20 --ewma-span 30 \
            --cost-bps 20 --max-gross 1.0 \
            --output results/crypto_combined_eqw \
            -v 2>&1

      - name: Crypto walk-forward with risk model
        run: |
          python scripts/run_combined.py \
            --tickers BTC-USD,ETH-USD \
            --start 2018-01-01 --end 2023-12-31 \
            --signals momentum,mean_reversion,ewma_momentum,volatility \
            --mode walk_forward \
            --train-days 504 --test-days 252 --step-days 252 \
            --weight-step 0.25 \
            --lookback 20 --ewma-span 30 \
            --cost-bps 20 --max-gross 1.0 \
            --risk-model \
            --clip-zscore 2.0 --smooth-halflife 5 \
            --vol-target 0.15 --max-leverage 2.0 \
            --output results/crypto_combined_wf_risk \
            -v 2>&1

      - name: Summarise all results
        shell: bash
        run: |
          python - <<'PY'
          import json, pathlib

          root = pathlib.Path("results")
          print(f"{'Strategy':<40} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'AnnVol':>8}")
          print("-" * 80)
          for m in sorted(root.rglob("metrics.json")):
              d = json.loads(m.read_text())
              name = str(m.parent.relative_to(root))
              metrics = d.get("overall_metrics", d)
              sharpe_key = "oos_sharpe" if "oos_sharpe" in metrics else "sharpe"
              cagr_key = "oos_cagr" if "oos_cagr" in metrics else "cagr"
              dd_key = "oos_max_dd" if "oos_max_dd" in metrics else "max_dd"
              vol_key = "oos_ann_vol" if "oos_ann_vol" in metrics else "ann_vol"
              sharpe = metrics.get(sharpe_key, float("nan"))
              cagr = metrics.get(cagr_key, float("nan"))
              max_dd = metrics.get(dd_key, float("nan"))
              ann_vol = metrics.get(vol_key, float("nan"))
              print(f"{name:<40} {sharpe:>8.3f} {cagr:>7.2%} {max_dd:>7.2%} {ann_vol:>7.2%}")
          PY

      - name: Commit results
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          if [ -d results ]; then
            git add results
            git commit -m "Update backtest results: residual momentum + 12-1 on 40-stock universe [skip ci]" || echo "No changes"
            git push || true
          else
            echo "No results directory found; skipping commit."
          fi
YMLEOF
echo "Created combined_backtests.yml"

# ----------------------------------------------------------------
# 8. Validate
# ----------------------------------------------------------------
echo ""
echo "=== Running validation ==="
pip install -e ".[dev,arima]" --quiet 2>/dev/null
ruff check --fix src/ tests/ scripts/
echo "Ruff: OK"
pytest -v --tb=short
echo ""
echo "=== All changes applied. Ready to commit. ==="
echo ""
echo "Run:"
echo '  git add -A'
echo '  git commit -m "Add risk model, residual momentum, skip-month momentum, broad 40-stock universe"'
echo '  git push origin test'
