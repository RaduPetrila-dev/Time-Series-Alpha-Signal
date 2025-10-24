from __future__ import annotations

from typing import Any
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt

from .data import load_synthetic_prices, load_csv_prices, load_yfinance_prices
from .backtest import backtest

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
     elif args.tickers is not None:
         tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
         prices = load_yfinance_prices(
             tickers=tickers,
             start=args.start_date,
             end=args.end_date,
             interval=args.interval,
         )
         if prices.empty:
             print("ERROR: No data downloaded from Yahoo Finance for the given tickers/date range.", file=sys.stderr)
             sys.exit(2)

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
    run.set_defaults(func=cmd_run)
    # example subcommand
    ex = sub.add_parser("example", help="Print example config")
    ex.set_defaults(func=cmd_print_example)
    return p


def main() -> None:
    """Entry point for command‑line execution."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

run.add_argument("--no-plots", action="store_true", help="Skip saving plots (CI-safe).")

(out_dir / "metrics.json").write_text(json.dumps(out["metrics"], indent=2))

if not args.no_plots:
    fig1 = plt.figure()
    out["equity"].plot(title="Equity Curve")
    fig1.savefig(out_dir / "equity.png", bbox_inches="tight")
    plt.close(fig1)

    dd = (out["equity"] / out["equity"].cummax() - 1)
    fig2 = plt.figure()
    dd.plot(title="Drawdown")
    fig2.savefig(out_dir / "drawdown.png", bbox_inches="tight")
    plt.close(fig2)


