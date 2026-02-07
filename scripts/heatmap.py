#!/usr/bin/env python
"""Parameter sweep for signal backtests.

Run a grid of lookback windows and gross exposure values for a given
signal type and visualise the resulting Sharpe ratios as a heatmap.

Usage
-----
::

    # Synthetic data, default grid
    python scripts/heatmap.py

    # Custom grid on real data
    python scripts/heatmap.py --csv data/prices_faang.csv \\
        --signal momentum \\
        --lookbacks 5 10 20 40 60 \\
        --grosses 0.5 1.0 1.5 2.0 \\
        --output sweep_results

    # Include rebalance and cost model
    python scripts/heatmap.py --rebalance weekly --impact-model sqrt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from time_series_alpha_signal.backtest import BacktestResult, backtest
from time_series_alpha_signal.data import load_csv_prices, load_synthetic_prices

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------


def run_sweep(
    prices: "pd.DataFrame",
    lookbacks: Sequence[int],
    grosses: Sequence[float],
    **kwargs: Any,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    """Run a grid of backtests over lookback x gross exposure.

    Parameters
    ----------
    prices : DataFrame
        Price data.
    lookbacks : sequence of int
        Lookback windows to test.
    grosses : sequence of float
        Gross exposure limits to test.
    **kwargs
        Forwarded to :func:`backtest` (e.g. ``signal_type``,
        ``cost_bps``, ``rebalance``, ``impact_model``).

    Returns
    -------
    sharpes : ndarray
        2-D array of Sharpe ratios, shape ``(len(lookbacks), len(grosses))``.
    details : nested list of dict
        Full metrics for each combination.
    """
    n_rows = len(lookbacks)
    n_cols = len(grosses)
    sharpes = np.zeros((n_rows, n_cols))
    details: list[list[dict[str, Any]]] = [
        [{} for _ in range(n_cols)] for _ in range(n_rows)
    ]

    total = n_rows * n_cols
    count = 0

    for i, lb in enumerate(lookbacks):
        for j, gross in enumerate(grosses):
            count += 1
            logger.info(
                "Sweep %d/%d: lookback=%d, max_gross=%.2f",
                count,
                total,
                lb,
                gross,
            )

            result: BacktestResult = backtest(
                prices, lookback=lb, max_gross=gross, **kwargs
            )

            sharpe = result.metrics.get("sharpe", float("nan"))
            sharpes[i, j] = sharpe
            details[i][j] = result.metrics

    return sharpes, details


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_heatmap(
    matrix: np.ndarray,
    lookbacks: Sequence[int],
    grosses: Sequence[float],
    out_path: Path,
    signal_type: str = "",
) -> None:
    """Save an annotated heatmap of Sharpe ratios.

    Parameters
    ----------
    matrix : ndarray
        2-D Sharpe ratios.
    lookbacks : sequence of int
        Row labels.
    grosses : sequence of float
        Column labels.
    out_path : Path
        Output directory.
    signal_type : str
        Signal name for the title.
    """
    fig, ax = plt.subplots(
        figsize=(2.0 + 1.0 * len(grosses), 1.5 + 0.7 * len(lookbacks))
    )

    # Color map: diverging around zero
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.5)
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=-vmax,
        vmax=vmax,
    )

    # Annotate each cell with the Sharpe value
    for i in range(len(lookbacks)):
        for j in range(len(grosses)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
            )

    ax.set_xticks(range(len(grosses)))
    ax.set_yticks(range(len(lookbacks)))
    ax.set_xticklabels([f"{g:.2f}" for g in grosses])
    ax.set_yticklabels([str(lb) for lb in lookbacks])
    ax.set_xlabel("Gross Exposure")
    ax.set_ylabel("Lookback Window")

    title = "Sharpe Ratio: Lookback vs Gross Exposure"
    if signal_type:
        title = f"{signal_type} — {title}"
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="Sharpe Ratio")
    fig.tight_layout()

    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s/heatmap.png", out_path)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_csv(
    details: list[list[dict[str, Any]]],
    lookbacks: Sequence[int],
    grosses: Sequence[float],
    out_path: Path,
) -> None:
    """Write a flat CSV of all sweep metrics.

    Parameters
    ----------
    details : nested list of dict
        Metrics from :func:`run_sweep`.
    lookbacks : sequence of int
        Lookback values.
    grosses : sequence of float
        Gross exposure values.
    out_path : Path
        Output directory.
    """
    import pandas as pd

    rows = []
    for i, lb in enumerate(lookbacks):
        for j, gross in enumerate(grosses):
            row = {"lookback": lb, "max_gross": gross, **details[i][j]}
            rows.append(row)

    if not rows:
        logger.warning("No sweep results to write.")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_path / "sweep_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved metrics CSV to %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep: lookback x gross exposure heatmap.",
    )

    # Signal
    parser.add_argument(
        "--signal",
        type=str,
        default="momentum",
        choices=_SIGNAL_CHOICES,
        help="Signal type (default: momentum).",
    )

    # Grid
    parser.add_argument(
        "--lookbacks",
        type=int,
        nargs="*",
        default=[5, 10, 20, 40, 60],
        help="Lookback windows to test (default: 5 10 20 40 60).",
    )
    parser.add_argument(
        "--grosses",
        type=float,
        nargs="*",
        default=[0.5, 1.0, 1.5, 2.0],
        help="Gross exposure limits (default: 0.5 1.0 1.5 2.0).",
    )

    # Backtest parameters
    parser.add_argument(
        "--cost-bps",
        dest="cost_bps",
        type=float,
        default=10.0,
        help="Transaction cost in bps (default: 10).",
    )
    parser.add_argument(
        "--max-leverage",
        dest="max_leverage",
        type=float,
        default=None,
        help="Maximum gross leverage (optional).",
    )
    parser.add_argument(
        "--max-drawdown",
        dest="max_drawdown",
        type=float,
        default=None,
        help="Drawdown stop fraction (optional).",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default="daily",
        choices=["daily", "weekly", "monthly"],
        help="Rebalance frequency (default: daily).",
    )
    parser.add_argument(
        "--impact-model",
        dest="impact_model",
        type=str,
        default="proportional",
        choices=["proportional", "sqrt"],
        help="Cost model (default: proportional).",
    )

    # Data source
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV file with price data.",
    )
    parser.add_argument(
        "--n-names",
        dest="n_names",
        type=int,
        default=20,
        help="Synthetic assets (default: 20).",
    )
    parser.add_argument(
        "--n-days",
        dest="n_days",
        type=int,
        default=750,
        help="Synthetic days (default: 750).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data.",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="sweep",
        help="Output directory (default: sweep).",
    )

    args = parser.parse_args()

    # Load prices
    if args.csv:
        prices = load_csv_prices(args.csv)
    else:
        prices = load_synthetic_prices(
            n_names=args.n_names, n_days=args.n_days, seed=args.seed
        )

    logger.info(
        "Sweep: signal=%s, %d lookbacks x %d grosses = %d combinations.",
        args.signal,
        len(args.lookbacks),
        len(args.grosses),
        len(args.lookbacks) * len(args.grosses),
    )

    # Run sweep
    sharpe_matrix, details = run_sweep(
        prices,
        lookbacks=args.lookbacks,
        grosses=args.grosses,
        signal_type=args.signal,
        cost_bps=args.cost_bps,
        max_leverage=args.max_leverage,
        max_drawdown=args.max_drawdown,
        rebalance=args.rebalance,
        impact_model=args.impact_model,
    )

    out_path = Path(args.output).resolve()

    # Outputs
    plot_heatmap(
        sharpe_matrix, args.lookbacks, args.grosses, out_path,
        signal_type=args.signal,
    )
    write_csv(details, args.lookbacks, args.grosses, out_path)

    # JSON dump
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "sweep.json").write_text(json.dumps(details, indent=2))

    # Summary to stdout
    best_idx = np.unravel_index(np.nanargmax(sharpe_matrix), sharpe_matrix.shape)
    best_lb = args.lookbacks[best_idx[0]]
    best_gross = args.grosses[best_idx[1]]
    best_sharpe = sharpe_matrix[best_idx]

    summary = {
        "best_lookback": best_lb,
        "best_gross": best_gross,
        "best_sharpe": round(float(best_sharpe), 4),
        "grid_size": len(args.lookbacks) * len(args.grosses),
    }
    print(json.dumps(summary, indent=2))
    logger.info("Best: lookback=%d, gross=%.2f, Sharpe=%.3f", best_lb, best_gross, best_sharpe)


if __name__ == "__main__":
    main()
