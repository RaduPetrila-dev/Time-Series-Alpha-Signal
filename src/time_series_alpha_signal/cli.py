"""Command-line interface for the Time-Series Alpha Signal framework.

Subcommands
-----------
``run``
    Execute a single backtest on synthetic, CSV, or Yahoo Finance data.
``cv``
    Estimate an out-of-sample Sharpe ratio distribution via purged
    (or combinatorial) cross-validation.
``train``
    Train logistic-regression meta-models per asset using triple-barrier
    labels and report cross-validated accuracy.
``example``
    Print an example configuration for the ``run`` command.

All subcommands write JSON metrics and (optionally) plots to the
specified output directory.

.. note::
   ``matplotlib`` is forced to the ``Agg`` backend at import time so
   the CLI works in headless / CI environments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Force non-interactive backend before any pyplot import.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult, backtest  # noqa: E402
from .data import load_csv_prices, load_synthetic_prices, load_yfinance_prices  # noqa: E402
from .evaluation import cross_validate_sharpe  # noqa: E402
from .models import train_meta_model  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal choices (kept in sync with the signal registry)
# ---------------------------------------------------------------------------

_SIGNAL_CHOICES = [
    "momentum",
    "mean_reversion",
    "arima",
    "volatility",
    "vol_scaled_momentum",
    "regime_switch",
    "ewma_momentum",
    "ma_crossover",
]

_REBALANCE_CHOICES = ["daily", "weekly", "monthly"]
_IMPACT_CHOICES = ["proportional", "sqrt"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_prices_for_cli(args: argparse.Namespace) -> pd.DataFrame:
    """Load prices from CSV, Yahoo Finance, or the synthetic generator.

    The three data sources are mutually exclusive.  ``--csv`` takes
    priority, then ``--tickers``, and finally the synthetic fallback.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments.

    Returns
    -------
    DataFrame
        Price data indexed by datetime with one column per asset.
    """
    if getattr(args, "csv", None) is not None:
        logger.info("Loading prices from CSV: %s", args.csv)
        return load_csv_prices(args.csv)

    if getattr(args, "tickers", None) is not None:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        logger.info("Fetching prices from yfinance: %s", tickers)
        return load_yfinance_prices(
            tickers=tickers,
            start=getattr(args, "start_date", None),
            end=getattr(args, "end_date", None),
            interval=getattr(args, "interval", "1d"),
        )

    names = getattr(args, "names", 20)
    days = getattr(args, "days", 750)
    seed = getattr(args, "seed", 42)
    logger.info(
        "Generating synthetic prices: %d assets, %d days, seed=%d",
        names,
        days,
        seed,
    )
    return load_synthetic_prices(n_names=names, n_days=days, seed=seed)


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Attach the shared dataset arguments to *parser*.

    Centralises ``--csv``, ``--tickers``, ``--start-date``,
    ``--end-date``, and ``--interval`` so every subcommand gets
    them without duplication.
    """
    group = parser.add_argument_group("data source")
    group.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a CSV file with price data (datetime index, asset columns).",
    )
    group.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers for yfinance (e.g. AAPL,MSFT,GOOGL).",
    )
    group.add_argument(
        "--start-date",
        dest="start_date",
        type=str,
        default=None,
        help="Start date for yfinance data (YYYY-MM-DD).",
    )
    group.add_argument(
        "--end-date",
        dest="end_date",
        type=str,
        default=None,
        help="End date for yfinance data (YYYY-MM-DD).",
    )
    group.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Sampling interval for yfinance data (default: 1d).",
    )


def _add_signal_args(parser: argparse.ArgumentParser) -> None:
    """Attach signal-related arguments to *parser*."""
    group = parser.add_argument_group("signal parameters")
    group.add_argument(
        "--signal",
        type=str,
        default="momentum",
        choices=_SIGNAL_CHOICES,
        help="Signal type (default: momentum).",
    )
    group.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Lookback window for simple signals (default: 20).",
    )
    group.add_argument(
        "--arima-order",
        dest="arima_order",
        nargs=3,
        type=int,
        default=[1, 0, 1],
        metavar=("P", "D", "Q"),
        help="ARIMA order for the arima signal (default: 1 0 1).",
    )
    group.add_argument(
        "--vol-window",
        dest="vol_window",
        type=int,
        default=20,
        help="Rolling volatility window for vol-based signals (default: 20).",
    )
    group.add_argument(
        "--vol-threshold",
        dest="vol_threshold",
        type=float,
        default=0.02,
        help="Volatility threshold for regime_switch (default: 0.02).",
    )
    group.add_argument(
        "--ewma-span",
        dest="ewma_span",
        type=int,
        default=20,
        help="EWMA span for ewma_momentum (default: 20).",
    )
    group.add_argument(
        "--ma-short",
        dest="ma_short",
        type=int,
        default=10,
        help="Short MA window for ma_crossover (default: 10).",
    )
    group.add_argument(
        "--ma-long",
        dest="ma_long",
        type=int,
        default=50,
        help="Long MA window for ma_crossover (default: 50).",
    )


def _save_plots(result: BacktestResult, out_dir: Path) -> None:
    """Write equity curve and drawdown plots to *out_dir*.

    Parameters
    ----------
    result : BacktestResult
        Completed backtest output.
    out_dir : Path
        Directory to write PNG files into.
    """
    # Equity curve
    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity.plot(ax=ax, title="Equity Curve")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("")
    fig.savefig(out_dir / "equity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Drawdown
    dd = result.equity / result.equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    dd.plot(ax=ax, title="Drawdown", color="crimson")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.fill_between(dd.index, dd.values, alpha=0.3, color="crimson")
    fig.savefig(out_dir / "drawdown.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Rolling Sharpe (63-day, ~1 quarter)
    rolling_sharpe = (
        result.daily.rolling(63).mean()
        / result.daily.rolling(63).std()
        * np.sqrt(252)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    rolling_sharpe.plot(ax=ax, title="Rolling 63-day Sharpe Ratio")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("")
    fig.savefig(out_dir / "rolling_sharpe.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Monthly returns heatmap
    try:
        monthly = result.daily.copy()
        monthly.index = pd.to_datetime(monthly.index)
        monthly_ret = monthly.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        pivot = pd.DataFrame(
            {
                "year": monthly_ret.index.year,
                "month": monthly_ret.index.month,
                "return": monthly_ret.values,
            }
        )
        heatmap_data = pivot.pivot(index="year", columns="month", values="return")
        heatmap_data.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][: len(heatmap_data.columns)]

        fig, ax = plt.subplots(figsize=(12, max(3, len(heatmap_data) * 0.6)))
        im = ax.imshow(
            heatmap_data.values,
            cmap="RdYlGn",
            aspect="auto",
            vmin=-0.10,
            vmax=0.10,
        )
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index)
        ax.set_title("Monthly Returns Heatmap")
        fig.colorbar(im, ax=ax, label="Return")
        fig.savefig(out_dir / "monthly_heatmap.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception:
        logger.debug("Skipping monthly heatmap (insufficient data).", exc_info=True)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single backtest and save results.

    Loads prices from CSV, Yahoo Finance, or the synthetic generator,
    executes the backtest with the selected signal and parameters, then
    writes metrics JSON and (optionally) diagnostic plots to the output
    directory.
    """
    prices = _load_prices_for_cli(args)

    result: BacktestResult = backtest(
        prices=prices,
        lookback=args.lookback,
        max_gross=args.max_gross,
        cost_bps=args.cost_bps,
        signal_type=args.signal,
        arima_order=tuple(args.arima_order),
        max_leverage=args.max_leverage,
        max_drawdown=args.max_drawdown,
        vol_window=args.vol_window,
        vol_threshold=args.vol_threshold,
        ewma_span=args.ewma_span,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        rebalance=args.rebalance,
        impact_model=args.impact_model,
    )

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics
    (out_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2)
    )

    # Write daily returns and weights for reproducibility
    result.daily.to_csv(out_dir / "daily_returns.csv", header=True)
    result.weights.to_csv(out_dir / "weights.csv")

    # Plots
    if not getattr(args, "no_plots", False):
        _save_plots(result, out_dir)

    if result.stopped:
        logger.warning("Drawdown stop was triggered during this backtest.")

    print(json.dumps(result.metrics, indent=2))


def cmd_cv(args: argparse.Namespace) -> None:
    """Estimate Sharpe ratio distribution via purged cross-validation.

    Runs the full signal + backtest pipeline, then applies purged
    k-fold (or combinatorial) CV on the daily returns to produce
    a distribution of out-of-sample Sharpe ratios.
    """
    prices = _load_prices_for_cli(args)

    signal_kwargs: dict[str, Any] = {
        "vol_window": args.vol_window,
        "vol_threshold": args.vol_threshold,
        "arima_order": tuple(args.arima_order),
        "ewma_span": args.ewma_span,
        "ma_short": args.ma_short,
        "ma_long": args.ma_long,
    }

    sharpe_values = cross_validate_sharpe(
        prices=prices,
        signal_type=args.signal,
        lookback=args.lookback,
        max_gross=args.max_gross,
        cost_bps=args.cost_bps,
        n_splits=args.n_splits,
        embargo_pct=args.embargo_pct,
        combinatorial=getattr(args, "combinatorial", False),
        test_fold_size=getattr(args, "test_fold_size", 1),
        **signal_kwargs,
    )

    if len(sharpe_values) > 0:
        summary: dict[str, Any] = {
            "sharpe_values": sharpe_values,
            "mean": float(np.mean(sharpe_values)),
            "std": float(np.std(sharpe_values, ddof=0)),
            "median": float(np.median(sharpe_values)),
            "min": float(np.min(sharpe_values)),
            "max": float(np.max(sharpe_values)),
            "n_folds": len(sharpe_values),
        }
    else:
        summary = {
            "sharpe_values": [],
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_folds": 0,
        }

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cv_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    """Train meta-models per asset and report CV accuracy.

    Iterates over each asset, constructs momentum features and
    triple-barrier meta-labels, fits a logistic regression model
    with purged k-fold CV, and reports classification accuracy.
    """
    prices = _load_prices_for_cli(args)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    asset_metrics: dict[str, Any] = {}
    for col in prices.columns:
        logger.info("Training meta-model for %s", col)
        series = prices[col].astype(float)
        metrics = train_meta_model(
            prices=series,
            lookback=args.lookback,
            vol_span=args.vol_span,
            pt_sl=(args.pt_mult, args.sl_mult),
            horizon=args.horizon,
            n_splits=args.n_splits,
            embargo_pct=args.embargo_pct,
        )
        asset_metrics[col] = metrics

    valid_values = [
        m["cv_accuracy_mean"]
        for m in asset_metrics.values()
        if not np.isnan(m["cv_accuracy_mean"])
    ]
    if len(valid_values) > 0:
        overall_mean = float(np.mean(valid_values))
        overall_std = float(np.std(valid_values, ddof=0))
    else:
        overall_mean = float("nan")
        overall_std = float("nan")

    summary: dict[str, Any] = {
        "asset_metrics": asset_metrics,
        "overall_mean_accuracy": overall_mean,
        "overall_std_accuracy": overall_std,
        "n_assets": len(asset_metrics),
    }

    (out_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_print_example(args: argparse.Namespace) -> None:
    """Print an example configuration for the ``run`` subcommand."""
    example = {
        "lookback": 20,
        "max_gross": 1.0,
        "cost_bps": 10.0,
        "names": 20,
        "days": 750,
        "output": "results",
        "signal": "momentum",
        "rebalance": "daily",
        "impact_model": "proportional",
    }
    print(json.dumps(example, indent=2))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="tsalpha",
        description="Time-Series Alpha Signal CLI",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- run -------------------------------------------------------------
    run = sub.add_parser("run", help="Run a single backtest.")
    _add_dataset_args(run)
    _add_signal_args(run)

    run_bt = run.add_argument_group("backtest parameters")
    run_bt.add_argument(
        "--max-gross",
        dest="max_gross",
        type=float,
        default=1.0,
        help="Gross exposure limit (default: 1.0).",
    )
    run_bt.add_argument(
        "--cost-bps",
        dest="cost_bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points (default: 10).",
    )
    run_bt.add_argument(
        "--max-leverage",
        dest="max_leverage",
        type=float,
        default=None,
        help="Maximum gross leverage (optional).",
    )
    run_bt.add_argument(
        "--max-drawdown",
        dest="max_drawdown",
        type=float,
        default=None,
        help="Drawdown stop as fraction (e.g. 0.2 for -20%%).",
    )
    run_bt.add_argument(
        "--rebalance",
        type=str,
        default="daily",
        choices=_REBALANCE_CHOICES,
        help="Rebalance frequency (default: daily).",
    )
    run_bt.add_argument(
        "--impact-model",
        dest="impact_model",
        type=str,
        default="proportional",
        choices=_IMPACT_CHOICES,
        help="Transaction cost model (default: proportional).",
    )

    run_out = run.add_argument_group("output")
    run_out.add_argument(
        "--names",
        type=int,
        default=20,
        help="Number of synthetic assets (default: 20).",
    )
    run_out.add_argument(
        "--days",
        type=int,
        default=750,
        help="Number of synthetic days (default: 750).",
    )
    run_out.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation.",
    )
    run_out.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results).",
    )
    run_out.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (useful in CI).",
    )
    run.set_defaults(func=cmd_run)

    # -- cv --------------------------------------------------------------
    cv = sub.add_parser("cv", help="Estimate out-of-sample Sharpe via purged CV.")
    _add_dataset_args(cv)
    _add_signal_args(cv)

    cv_params = cv.add_argument_group("cross-validation parameters")
    cv_params.add_argument(
        "--max-gross",
        dest="max_gross",
        type=float,
        default=1.0,
        help="Gross exposure limit (default: 1.0).",
    )
    cv_params.add_argument(
        "--cost-bps",
        dest="cost_bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points (default: 10).",
    )
    cv_params.add_argument(
        "--n-splits",
        dest="n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    cv_params.add_argument(
        "--embargo-pct",
        dest="embargo_pct",
        type=float,
        default=0.0,
        help="Embargo fraction between folds (default: 0.0).",
    )
    cv_params.add_argument(
        "--combinatorial",
        action="store_true",
        help="Use combinatorial purged CV instead of standard k-fold.",
    )
    cv_params.add_argument(
        "--test-fold-size",
        dest="test_fold_size",
        type=int,
        default=1,
        help="Test fold size for combinatorial CV (default: 1).",
    )

    cv_out = cv.add_argument_group("output")
    cv_out.add_argument(
        "--names",
        type=int,
        default=20,
        help="Number of synthetic assets (default: 20).",
    )
    cv_out.add_argument(
        "--days",
        type=int,
        default=750,
        help="Number of synthetic days (default: 750).",
    )
    cv_out.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data.",
    )
    cv_out.add_argument(
        "--output",
        type=str,
        default="results_cv",
        help="Output directory (default: results_cv).",
    )
    cv.set_defaults(func=cmd_cv)

    # -- train -----------------------------------------------------------
    train = sub.add_parser(
        "train", help="Train and evaluate a predictive meta-model."
    )
    _add_dataset_args(train)

    train_params = train.add_argument_group("training parameters")
    train_params.add_argument(
        "--lookback",
        type=int,
        default=20,
        help="Lookback window for momentum feature (default: 20).",
    )
    train_params.add_argument(
        "--vol-span",
        dest="vol_span",
        type=int,
        default=50,
        help="Span for volatility estimate (default: 50).",
    )
    train_params.add_argument(
        "--pt-mult",
        dest="pt_mult",
        type=float,
        default=1.0,
        help="Profit-taking multiplier for triple barrier (default: 1.0).",
    )
    train_params.add_argument(
        "--sl-mult",
        dest="sl_mult",
        type=float,
        default=1.0,
        help="Stop-loss multiplier for triple barrier (default: 1.0).",
    )
    train_params.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Vertical barrier horizon in days (default: 5).",
    )
    train_params.add_argument(
        "--n-splits",
        dest="n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    train_params.add_argument(
        "--embargo-pct",
        dest="embargo_pct",
        type=float,
        default=0.0,
        help="Embargo fraction between folds (default: 0.0).",
    )

    train_out = train.add_argument_group("output")
    train_out.add_argument(
        "--names",
        type=int,
        default=20,
        help="Number of synthetic assets (default: 20).",
    )
    train_out.add_argument(
        "--days",
        type=int,
        default=750,
        help="Number of synthetic days (default: 750).",
    )
    train_out.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data.",
    )
    train_out.add_argument(
        "--output",
        type=str,
        default="results_train",
        help="Output directory (default: results_train).",
    )
    train.set_defaults(func=cmd_train)

    # -- example ---------------------------------------------------------
    ex = sub.add_parser("example", help="Print example config.")
    ex.set_defaults(func=cmd_print_example)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python -m time_series_alpha_signal.cli``."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging based on --verbose flag
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=level,
        stream=sys.stderr,
    )

    args.func(args)
