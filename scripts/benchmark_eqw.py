"""Generate benchmark equity curves and metrics for strategy comparison.

This script computes passive benchmark portfolios against which the
active signal strategies are evaluated.  Three benchmark types are
supported:

* **equal-weight** (``eqw``) -- equal allocation across all assets,
  rebalanced daily.
* **buy-and-hold** (``bh``) -- buy equal amounts on day one, never
  rebalance.  Weights drift with price changes.
* **inverse-volatility** (``ivol``) -- weight each asset by the
  inverse of its trailing 20-day volatility (risk-parity lite).

Usage
-----
::

    python scripts/benchmark_eqw.py --csv data/prices_faang_2018_2023.csv
    python scripts/benchmark_eqw.py --csv data/prices.csv --type ivol --output results/benchmark

The script writes two files to the output directory:

* ``{type}_equity.csv`` -- daily cumulative equity curve.
* ``{type}_metrics.json`` -- summary statistics (CAGR, Sharpe,
  max drawdown, annualised volatility, Calmar ratio).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute daily returns for an equal-weight portfolio.

    Parameters
    ----------
    returns : DataFrame
        Daily asset returns.

    Returns
    -------
    Series
        Daily portfolio returns (simple average across assets).
    """
    return returns.mean(axis=1)


def buy_and_hold_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute daily returns for a buy-and-hold portfolio.

    Weights start equal on day one and drift with cumulative returns.
    No rebalancing occurs.

    Parameters
    ----------
    returns : DataFrame
        Daily asset returns.

    Returns
    -------
    Series
        Daily portfolio returns.
    """
    # Cumulative wealth per asset (starting at 1.0)
    wealth = (1 + returns).cumprod()
    # Portfolio weight = asset wealth / total wealth
    total = wealth.sum(axis=1)
    weights = wealth.div(total, axis=0)
    # Portfolio return = sum(weight_prev * return_today)
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    return port_ret


def inverse_vol_returns(
    returns: pd.DataFrame,
    vol_window: int = 20,
) -> pd.Series:
    """Compute daily returns for an inverse-volatility portfolio.

    Each asset is weighted by the inverse of its trailing rolling
    volatility, normalised to sum to 1.

    Parameters
    ----------
    returns : DataFrame
        Daily asset returns.
    vol_window : int, default 20
        Rolling window for volatility estimation.

    Returns
    -------
    Series
        Daily portfolio returns.
    """
    vol = returns.rolling(vol_window).std()
    inv_vol = 1.0 / (vol + 1e-8)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    return port_ret


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_benchmark_metrics(
    daily: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute summary statistics for a benchmark return series.

    Parameters
    ----------
    daily : Series
        Daily portfolio returns.
    periods_per_year : int, default 252
        Annualisation factor.

    Returns
    -------
    dict
        CAGR, annualised volatility, Sharpe ratio, max drawdown,
        and Calmar ratio.
    """
    r = daily.dropna()
    n = len(r)
    if n < 2:
        return {
            "cagr": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "calmar": float("nan"),
            "n_days": n,
        }

    equity = (1 + r).cumprod()
    final = float(equity.iloc[-1])

    cagr = float(final ** (periods_per_year / n) - 1) if final > 0 else -1.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    vol = r.std(ddof=1)
    sharpe = float(np.sqrt(periods_per_year) * r.mean() / vol) if vol > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "cagr": round(cagr, 6),
        "ann_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 6),
        "calmar": round(calmar, 4),
        "n_days": n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_BENCHMARK_FUNCS = {
    "eqw": equal_weight_returns,
    "bh": buy_and_hold_returns,
    "ivol": inverse_vol_returns,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark equity curves and metrics.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV with price data (datetime index, asset columns).",
    )
    parser.add_argument(
        "--date-col",
        dest="date_col",
        type=str,
        default="Date",
        help="Name of the date column in the CSV (default: Date).",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="eqw",
        choices=list(_BENCHMARK_FUNCS.keys()),
        help="Benchmark type: eqw (equal-weight), bh (buy-and-hold), "
        "ivol (inverse-vol). Default: eqw.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark",
        help="Output directory (default: results/benchmark).",
    )
    args = parser.parse_args()

    # Load prices
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("File not found: %s", csv_path)
        sys.exit(1)

    prices = pd.read_csv(csv_path, parse_dates=[args.date_col], index_col=args.date_col)
    prices = prices.select_dtypes(include=[np.number])
    logger.info(
        "Loaded %d rows x %d assets from %s.",
        len(prices),
        prices.shape[1],
        csv_path.name,
    )

    # Compute returns
    rets = prices.pct_change().dropna()

    # Build benchmark
    bench_func = _BENCHMARK_FUNCS[args.type]
    bench_returns = bench_func(rets)
    equity = (1 + bench_returns).cumprod()

    # Compute metrics
    metrics = compute_benchmark_metrics(bench_returns)
    logger.info(
        "Benchmark '%s': CAGR=%.2f%%, Sharpe=%.2f, MaxDD=%.2f%%",
        args.type,
        metrics["cagr"] * 100,
        metrics["sharpe"],
        metrics["max_dd"] * 100,
    )

    # Save outputs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_path = out_dir / f"{args.type}_equity.csv"
    equity.to_csv(equity_path, header=True)
    logger.info("Saved equity curve: %s", equity_path)

    metrics_path = out_dir / f"{args.type}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics: %s", metrics_path)

    # Print to stdout
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
