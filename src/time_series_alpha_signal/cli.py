from __future__ import annotations
from typing import Any
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

from .data import load_synthetic_prices
from .backtest import backtest


def cmd_run(args: argparse.Namespace) -> None:
    """Run a synthetic backtest based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing lookback, max_gross, cost_bps,
        number of names, number of days, random seed and output path.
    """
    # generate synthetic price data
    prices = load_synthetic_prices(n_names=args.names, n_days=args.days, seed=args.seed)
    # run backtest
    out = backtest(
        prices=prices,
        lookback=args.lookback,
        max_gross=args.max_gross,
        cost_bps=args.cost_bps,
        seed=args.seed,
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
    p = argparse.ArgumentParser(prog="tsalpha", description="Time-series alpha CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    # run subcommand
    run = sub.add_parser("run", help="Run synthetic backtest")
    run.add_argument("--lookback", type=int, default=20)
    run.add_argument("--max-gross", dest="max_gross", type=float, default=1.0)
    run.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0)
    run.add_argument("--names", type=int, default=20)
    run.add_argument("--days", type=int, default=750)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", type=str, default="results")
    run.set_defaults(func=cmd_run)
    # example subcommand
    ex = sub.add_parser("example", help="Print example config")
    ex.set_defaults(func=cmd_print_example)
    return p


def main() -> None:
    """Entry point for command-line execution."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
