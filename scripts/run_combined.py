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
    parser = argparse.ArgumentParser(description="Run combined signal backtests.")
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated tickers (e.g. AAPL,AMZN or BTC-USD,ETH-USD).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (overrides --tickers).",
    )
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument(
        "--signals",
        type=str,
        required=True,
        help="Comma-separated signal names.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="equal_weight",
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
        "--risk-model",
        dest="risk_model",
        action="store_true",
        default=False,
        help="Enable the risk model pipeline.",
    )
    parser.add_argument("--clip-zscore", dest="clip_zscore", type=float, default=2.0)
    parser.add_argument("--smooth-halflife", dest="smooth_halflife", type=int, default=5)
    parser.add_argument(
        "--no-inverse-vol",
        dest="inverse_vol",
        action="store_false",
        default=True,
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
