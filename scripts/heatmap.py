#!/usr/bin/env python
"""Parameter sweep for signal backtests.

Run a grid of lookback windows and gross exposure values for a given
signal type and visualise the resulting Sharpe ratios as a heatmap.
Results are saved to the specified output directory.  This script is
a lightly modified copy of the upstream heatmap utility and is
included here for completeness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from time_series_alpha_signal.data import load_synthetic_prices, load_csv_prices
from time_series_alpha_signal.backtest import backtest


def run_sweep(
    prices, lookbacks: Sequence[int], grosses: Sequence[float], **kwargs
) -> tuple[np.ndarray, list[list[dict[str, float]]]]:
    """Run a grid of backtests and collect Sharpe ratio and metrics.

    Parameters
    ----------
    prices : DataFrame
        Price data indexed by datetime.
    lookbacks : sequence of int
        Lookback windows to test.
    grosses : sequence of float
        Gross exposure limits to test.
    **kwargs
        Additional keyword arguments forwarded to :func:`backtest` (e.g.
        ``signal_type``, ``cost_bps``, ``max_leverage``, ``max_drawdown``).

    Returns
    -------
    tuple of (matrix, metrics)
        ``matrix`` is a 2‑D NumPy array of Sharpe ratios with shape
        (len(lookbacks), len(grosses)); ``metrics`` is a nested list of
        dictionaries with the full metric dictionary for each combination.
    """
    n_rows = len(lookbacks)
    n_cols = len(grosses)
    sharpes = np.zeros((n_rows, n_cols))
    details: list[list[dict[str, float]]] = [[{} for _ in grosses] for _ in lookbacks]
    for i, lb in enumerate(lookbacks):
        for j, gross in enumerate(grosses):
            result = backtest(prices, lookback=lb, max_gross=gross, **kwargs)
            sharpe = result["metrics"].get("sharpe", float("nan"))
            sharpes[i, j] = sharpe
            details[i][j] = result["metrics"]
    return sharpes, details


def plot_heatmap(matrix: np.ndarray, lookbacks: Sequence[int], grosses: Sequence[float], out_path: Path) -> None:
    """Save a heatmap of Sharpe ratios using matplotlib.

    Parameters
    ----------
    matrix : 2‑D array
        Sharpe ratios for each parameter combination.
    lookbacks : sequence of int
        Lookback values (rows of the matrix).
    grosses : sequence of float
        Gross exposure values (columns of the matrix).
    out_path : Path
        Directory where the heatmap PNG will be written.
    """
    fig, ax = plt.subplots(figsize=(1.5 + 0.5 * len(grosses), 1.5 + 0.5 * len(lookbacks)))
    cax = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(grosses)))
    ax.set_yticks(range(len(lookbacks)))
    ax.set_xticklabels([f"{g:.2f}" for g in grosses])
    ax.set_yticklabels([str(lb) for lb in lookbacks])
    ax.set_xlabel("Gross Exposure (max_gross)")
    ax.set_ylabel("Lookback Window")
    ax.set_title("Sharpe Ratio Heatmap")
    fig.colorbar(cax, ax=ax, label="Sharpe Ratio")
    fig.tight_layout()
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "heatmap.png", dpi=200)
    plt.close(fig)


def write_csv(details: list[list[dict[str, float]]], lookbacks: Sequence[int], grosses: Sequence[float], out_path: Path) -> None:
    """Write a CSV summary of metrics from the parameter sweep.

    Parameters
    ----------
    details : nested list of dict
        Full metric dictionaries from ``run_sweep``.
    lookbacks : sequence of int
        Lookback values.
    grosses : sequence of float
        Gross exposure values.
    out_path : Path
        Directory where the CSV file will be written.
    """
    # Flatten results into rows
    rows = []
    for i, lb in enumerate(lookbacks):
        for j, gross in enumerate(grosses):
            metric = details[i][j]
            row = {
                "lookback": lb,
                "max_gross": gross,
                **metric,
            }
            rows.append(row)
    # Determine header order
    headers = list(rows[0].keys()) if rows else []
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")


def main() -> None:
    """Entry point for the heatmap script."""
    parser = argparse.ArgumentParser(description="Run parameter sweep for time‑series alpha signals")
    parser.add_argument("--signal", type=str, default="momentum", choices=["momentum", "mean_reversion", "arima", "volatility"], help="Signal type")
    parser.add_argument("--lookbacks", type=int, nargs="*", default=[10, 20, 60], help="List of lookback windows to test")
    parser.add_argument("--grosses", type=float, nargs="*", default=[0.5, 1.0, 1.5], help="List of gross exposure limits to test")
    parser.add_argument("--cost-bps", dest="cost_bps", type=float, default=10.0, help="Transaction cost in basis points")
    parser.add_argument("--max-leverage", dest="max_leverage", type=float, default=None, help="Maximum gross leverage (optional)")
    parser.add_argument("--max-drawdown", dest="max_drawdown", type=float, default=None, help="Drawdown stop (optional)")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file of prices")
    parser.add_argument("--n-names", dest="n_names", type=int, default=20, help="Number of synthetic symbols (if no CSV)")
    parser.add_argument("--n-days", dest="n_days", type=int, default=750, help="Number of synthetic days (if no CSV)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic data")
    parser.add_argument("--output", type=str, default="sweep", help="Output directory for heatmap and metrics")
    args = parser.parse_args()
    # Load prices
    if args.csv:
        prices = load_csv_prices(args.csv)
    else:
        prices = load_synthetic_prices(n_names=args.n_names, n_days=args.n_days, seed=args.seed)
    # Run sweep
    sharpe_matrix, details = run_sweep(
        prices,
        lookbacks=args.lookbacks,
        grosses=args.grosses,
        signal_type=args.signal,
        cost_bps=args.cost_bps,
        max_leverage=args.max_leverage,
        max_drawdown=args.max_drawdown,
    )
    out_path = Path(args.output).resolve()
    # Plot heatmap and write CSV
    plot_heatmap(sharpe_matrix, args.lookbacks, args.grosses, out_path)
    write_csv(details, args.lookbacks, args.grosses, out_path)
    # Also dump JSON for convenience
    (out_path / "sweep.json").write_text(json.dumps(details, indent=2))
    # Print location of results
    print(f"Saved heatmap and metrics to {out_path}")


if __name__ == "__main__":
    main()
