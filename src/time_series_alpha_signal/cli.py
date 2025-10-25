from __future__ import annotations

from typing import Any
import argparse
import json
from pathlib import Path

# Always use a non‑interactive backend for matplotlib.  This prevents
# `RuntimeError: Invalid DISPLAY variable` when running in CI or other
# headless environments.  The backend must be set before importing
# `matplotlib.pyplot`.
import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import load_synthetic_prices, load_csv_prices, load_yfinance_prices
from .backtest import backtest
from .evaluation import cross_validate_sharpe
from .models import train_meta_model


def cmd_run(args: argparse.Namespace) -> None:
    """Run a synthetic or data‑driven backtest based on CLI args.

    This command loads price data from a CSV, Yahoo Finance or
    synthetic generator and runs the cross‑sectional backtest with the
    selected signal.  Results are saved to the specified output
    directory.  See :func:`time_series_alpha_signal.backtest.backtest`
    for parameter definitions.
    """
    # load price data based on user input: CSV, yfinance or synthetic
    if args.csv is not None:
        # load from local CSV
        prices = load_csv_prices(args.csv)
    elif args.tickers is not None:
        # load from Yahoo Finance using yfinance
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        prices = load_yfinance_prices(
            tickers=tickers,
            start=args.start_date,
            end=args.end_date,
            interval=args.interval,
        )
    else:
        # fallback: synthetic data
        prices = load_synthetic_prices(n_names=args.names, n_days=args.days, seed=args.seed)
    # run backtest with chosen signal and optional leverage/drawdown
    out = backtest(
        prices=prices,
        lookback=args.lookback,
        max_gross=args.max_gross,
        cost_bps=args.cost_bps,
        seed=args.seed,
        signal_type=args.signal,
        arima_order=tuple(args.arima_order),
        max_leverage=args.max_leverage,
        max_drawdown=args.max_drawdown,
        vol_window=args.vol_window,
        vol_threshold=args.vol_threshold,
        ewma_span=args.ewma_span,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
    )
    # prepare output directory
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # save metrics as JSON
    (out_dir / "metrics.json").write_text(json.dumps(out["metrics"], indent=2))
    # create plots unless suppressed
    if not getattr(args, "no_plots", False):
        # plot equity curve
        fig1 = plt.figure()
        out["equity"].plot(title="Equity Curve")
        fig1.savefig(out_dir / "equity.png", bbox_inches="tight")
        plt.close(fig1)
        # plot drawdown
        dd = (out["equity"] / out["equity"].cummax() - 1)
        fig2 = plt.figure()
        dd.plot(title="Drawdown")
        fig2.savefig(out_dir / "drawdown.png", bbox_inches="tight")
        plt.close(fig2)
    # print metrics to stdout for CLI consumption
    print(json.dumps(out["metrics"], indent=2))


def cmd_print_example(args: argparse.Namespace) -> None:
    """Print an example configuration for the run command."""
    example = {
        "lookback": 20,
        "max_gross": 1.0,
        "cost_bps": 10.0,
        "names": 20,
        "days": 750,
        "seed": 42,
        "output": "results",
    }
    print(json.dumps(example, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    p = argparse.ArgumentParser(prog="tsalpha", description="Time‑series alpha CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    # run subcommand
    run = sub.add_parser("run", help="Run a backtest")
    run.add_argument("--lookback", type=int, default=20, help="Lookback window for simple signals")
    run.add_argument("--max-gross", dest="max_gross", type=float, default=1.0, help="Gross exposure limit")
    run.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0, help="Transaction cost in bps")
    run.add_argument("--names", type=int, default=20, help="Number of synthetic symbols (if CSV not supplied)")
    run.add_argument("--days", type=int, default=750, help="Number of synthetic days (if CSV not supplied)")
    run.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    run.add_argument("--output", type=str, default="results", help="Output directory for results")
    run.add_argument(
        "--signal",
        type=str,
        default="momentum",
        choices=[
            "momentum",
            "mean_reversion",
            "arima",
            "volatility",
            "vol_scaled_momentum",
            "regime_switch",
            "ewma_momentum",
            "ma_crossover",
        ],
        help="Signal type to use",
    )
    run.add_argument("--arima-order", dest="arima_order", nargs=3, type=int, default=[1, 0, 1], metavar=("p", "d", "q"), help="ARIMA order for arima signal")
    run.add_argument("--max-leverage", dest="max_leverage", type=float, default=None, help="Maximum gross leverage (optional)")
    run.add_argument("--csv", type=str, default=None, help="Path to CSV file containing price data")
    run.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma‑separated list of tickers to fetch from yfinance (e.g. AAPL,MSFT,GOOGL)",
    )
    run.add_argument(
        "--start-date",
        dest="start_date",
        type=str,
        default=None,
        help="Start date for yfinance data in YYYY‑MM‑DD format",
    )
    run.add_argument(
        "--end-date",
        dest="end_date",
        type=str,
        default=None,
        help="End date for yfinance data in YYYY‑MM‑DD format",
    )
    run.add_argument(
        "--interval",
        dest="interval",
        type=str,
        default="1d",
        help="Sampling interval for yfinance data (e.g. 1d, 1wk)",
    )
    run.add_argument(
        "--max-drawdown",
        dest="max_drawdown",
        type=float,
        default=None,
        help="Optional drawdown stop (e.g. 0.2 for a 20% max drawdown); strategy goes flat after breach.",
    )
    run.add_argument(
        "--vol-window",
        dest="vol_window",
        type=int,
        default=20,
        help="Window length for volatility calculations used in vol_scaled_momentum and regime_switch",
    )
    run.add_argument(
        "--vol-threshold",
        dest="vol_threshold",
        type=float,
        default=0.02,
        help="Threshold for average realised volatility used in regime_switch",
    )
    # additional parameters for advanced signals
    run.add_argument(
        "--ewma-span",
        dest="ewma_span",
        type=int,
        default=20,
        help="Span parameter for EWMA momentum signal",
    )
    run.add_argument(
        "--ma-short",
        dest="ma_short",
        type=int,
        default=10,
        help="Short window length for moving average crossover signal",
    )
    run.add_argument(
        "--ma-long",
        dest="ma_long",
        type=int,
        default=50,
        help="Long window length for moving average crossover signal",
    )
    run.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save equity and drawdown plots (useful for CI)",
    )
    run.set_defaults(func=cmd_run)
    # example subcommand
    ex = sub.add_parser("example", help="Print example config")
    ex.set_defaults(func=cmd_print_example)

    # cross‑validation subcommand
    cv = sub.add_parser("cv", help="Estimate out‑of‑sample Sharpe via purged CV")
    cv.add_argument("--lookback", type=int, default=20, help="Lookback window for simple signals")
    cv.add_argument("--max-gross", dest="max_gross", type=float, default=1.0, help="Gross exposure limit")
    cv.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0, help="Transaction cost in bps")
    cv.add_argument("--signal", type=str, default="momentum", choices=[
        "momentum", "mean_reversion", "arima", "volatility", "vol_scaled_momentum",
        "regime_switch", "ewma_momentum", "ma_crossover",
    ], help="Signal type to use")
    cv.add_argument("--n-splits", dest="n_splits", type=int, default=5, help="Number of folds for CV")
    cv.add_argument("--embargo-pct", dest="embargo_pct", type=float, default=0.0, help="Embargo percentage between folds")
    cv.add_argument("--combinatorial", action="store_true", help="Use combinatorial purged CV")
    cv.add_argument("--test-fold-size", dest="test_fold_size", type=int, default=1, help="Test fold size for combinatorial CV")
    cv.add_argument("--output", type=str, default="results_cv", help="Output directory for CV results")
    # dataset options for cv
    cv.add_argument("--csv", type=str, default=None, help="Path to CSV file containing price data")
    cv.add_argument("--tickers", type=str, default=None, help="Comma‑separated list of tickers to fetch from yfinance")
    cv.add_argument("--start-date", dest="start_date", type=str, default=None, help="Start date for yfinance data in YYYY‑MM‑DD format")
    cv.add_argument("--end-date", dest="end_date", type=str, default=None, help="End date for yfinance data in YYYY‑MM‑DD format")
    cv.add_argument("--interval", dest="interval", type=str, default="1d", help="Sampling interval for yfinance data")
    # signal specific kwargs
    cv.add_argument("--vol-window", dest="vol_window", type=int, default=20, help="Volatility window for vol_scaled_momentum and regime_switch")
    cv.add_argument("--vol-threshold", dest="vol_threshold", type=float, default=0.02, help="Volatility threshold for regime_switch")
    cv.add_argument("--arima-order", dest="arima_order", nargs=3, type=int, default=[1, 0, 1], metavar=("p", "d", "q"), help="ARIMA order for arima signal")
    cv.add_argument("--ewma-span", dest="ewma_span", type=int, default=20, help="Span for EWMA momentum signal")
    cv.add_argument("--ma-short", dest="ma_short", type=int, default=10, help="Short window for MA crossover signal")
    cv.add_argument("--ma-long", dest="ma_long", type=int, default=50, help="Long window for MA crossover signal")
    cv.set_defaults(func=cmd_cv)

    # train subcommand for predictive meta‑labelling
    train = sub.add_parser("train", help="Train and evaluate a predictive meta‑model")
    train.add_argument("--lookback", type=int, default=20, help="Lookback window for momentum feature")
    train.add_argument("--vol-span", dest="vol_span", type=int, default=50, help="Span for volatility estimate")
    train.add_argument("--pt-mult", dest="pt_mult", type=float, default=1.0, help="Profit‑taking multiplier for triple barrier")
    train.add_argument("--sl-mult", dest="sl_mult", type=float, default=1.0, help="Stop‑loss multiplier for triple barrier")
    train.add_argument("--horizon", type=int, default=5, help="Vertical barrier horizon for triple barrier")
    train.add_argument("--n-splits", dest="n_splits", type=int, default=5, help="Number of folds for CV")
    train.add_argument("--embargo-pct", dest="embargo_pct", type=float, default=0.0, help="Embargo percentage between folds")
    train.add_argument("--output", type=str, default="results_train", help="Output directory for training results")
    # dataset options for train
    train.add_argument("--csv", type=str, default=None, help="Path to CSV file containing price data")
    train.add_argument("--tickers", type=str, default=None, help="Comma‑separated list of tickers to fetch from yfinance")
    train.add_argument("--start-date", dest="start_date", type=str, default=None, help="Start date for yfinance data in YYYY‑MM‑DD format")
    train.add_argument("--end-date", dest="end_date", type=str, default=None, help="End date for yfinance data in YYYY‑MM‑DD format")
    train.add_argument("--interval", dest="interval", type=str, default="1d", help="Sampling interval for yfinance data")
    train.set_defaults(func=cmd_train)
    return p


# --- New subcommand implementations ---

def _load_prices_for_cli(args: argparse.Namespace) -> pd.DataFrame:
    """Helper to load prices based on CLI arguments.

    This function encapsulates the logic for loading price data from a
    CSV, Yahoo Finance or synthetic generator based on mutually
    exclusive CLI options.  It is shared across multiple subcommands.

    Parameters
    ----------
    args : Namespace
        Parsed command‑line arguments containing the dataset options.

    Returns
    -------
    DataFrame
        Price data indexed by datetime with asset columns.
    """
    if getattr(args, "csv", None):
        return load_csv_prices(args.csv)
    if getattr(args, "tickers", None):
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        return load_yfinance_prices(
            tickers=tickers,
            start=args.start_date,
            end=args.end_date,
            interval=args.interval,
        )
    # synthetic fallback
    names = getattr(args, "names", 20)
    days = getattr(args, "days", 750)
    seed = getattr(args, "seed", 42)
    return load_synthetic_prices(n_names=names, n_days=days, seed=seed)


def cmd_cv(args: argparse.Namespace) -> None:
    """Estimate Sharpe ratio distribution via purged cross‑validation.

    This command runs a full backtest on the supplied price data and
    then applies purged k‑fold (or combinatorial) cross‑validation on
    the resulting daily returns to estimate a distribution of
    out‑of‑sample Sharpe ratios.  The distribution and summary
    statistics are saved as JSON in the output directory and printed
    to stdout.
    """
    import pandas as pd  # local import to avoid CLI import cycles
    # load prices
    prices = _load_prices_for_cli(args)
    # build signal kwargs from args
    signal_kwargs = {
        "vol_window": args.vol_window,
        "vol_threshold": args.vol_threshold,
        "arima_order": tuple(args.arima_order),
        "ewma_span": args.ewma_span,
        "ma_short": args.ma_short,
        "ma_long": args.ma_long,
    }
    # compute Sharpe ratios across folds
    sharpe_values = cross_validate_sharpe(
        prices=prices,
        signal_type=args.signal,
        lookback=args.lookback,
        max_gross=args.max_gross,
        cost_bps=args.cost_bps,
        n_splits=args.n_splits,
        embargo_pct=args.embargo_pct,
        combinatorial=args.combinatorial,
        test_fold_size=args.test_fold_size,
        **signal_kwargs,
    )
    # summarise distribution
    if len(sharpe_values) > 0:
        import numpy as np  # local import
        summary = {
            "sharpe_values": sharpe_values,
            "mean": float(np.mean(sharpe_values)),
            "std": float(np.std(sharpe_values, ddof=0)),
            "min": float(np.min(sharpe_values)),
            "max": float(np.max(sharpe_values)),
            "n_folds": len(sharpe_values),
        }
    else:
        summary = {
            "sharpe_values": [],
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_folds": 0,
        }
    # ensure output directory
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cv_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    """Train meta‑models on each asset and report CV accuracy.

    This command iterates over each asset in the price data, constructs
    momentum features and meta‑labels, fits a logistic regression model
    using purged k‑fold cross‑validation and reports the mean and
    standard deviation of the classification accuracy.  Results are
    saved per asset as well as aggregated across all assets.
    """
    import numpy as np  # local import to avoid module level dependency
    # load prices
    prices = _load_prices_for_cli(args)
    # prepare output directory
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_metrics = {}
    # iterate over columns (assets)
    for col in prices.columns:
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
    # aggregate metrics across assets
    valid_values = [m["cv_accuracy_mean"] for m in asset_metrics.values() if not np.isnan(m["cv_accuracy_mean"])]
    if len(valid_values) > 0:
        overall_mean = float(np.mean(valid_values))
        overall_std = float(np.std(valid_values, ddof=0))
    else:
        overall_mean = float("nan")
        overall_std = float("nan")
    summary = {
        "asset_metrics": asset_metrics,
        "overall_mean_accuracy": overall_mean,
        "overall_std_accuracy": overall_std,
        "n_assets": len(asset_metrics),
    }
    # write JSON file
    (out_dir / "train_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Entry point for command‑line execution."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
