"""Fetch historical price data from Yahoo Finance and save to CSV.

Usage
-----
::

    # Default: FAANG 2018-2023
    python scripts/fetch_data.py

    # Custom tickers and date range
    python scripts/fetch_data.py --tickers AAPL MSFT NVDA TSLA \\
        --start 2015-01-01 --end 2024-12-31 --output data/tech_prices.csv

    # Predefined universes
    python scripts/fetch_data.py --universe sp500_top10

The script writes a CSV with a datetime index and one column per
ticker, plus a companion ``_metadata.json`` file recording the
download parameters and data quality summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined ticker universes
# ---------------------------------------------------------------------------

UNIVERSES: dict[str, list[str]] = {
    "faang": ["AAPL", "AMZN", "GOOGL", "META", "NFLX"],
    "mag7": ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"],
    "sp500_top10": [
        "AAPL",
        "MSFT",
        "AMZN",
        "NVDA",
        "GOOGL",
        "META",
        "BRK-B",
        "LLY",
        "AVGO",
        "JPM",
    ],
    "sectors_etf": [
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "XLRE",
        "XLB",
        "XLC",
    ],
}


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Handles the yfinance API change where ``auto_adjust=True``
    (default in >= 0.2.31) returns ``"Close"`` instead of
    ``"Adj Close"``.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols.
    start : str
        Start date (YYYY-MM-DD), inclusive.
    end : str
        End date (YYYY-MM-DD), exclusive.
    interval : str, default "1d"
        Sampling interval.

    Returns
    -------
    DataFrame
        Prices indexed by datetime, one column per ticker.

    Raises
    ------
    ImportError
        If yfinance is not installed.
    ValueError
        If no data is returned.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required. Install with: pip install yfinance") from exc

    logger.info(
        "Downloading %d tickers: %s (start=%s, end=%s, interval=%s)",
        len(tickers),
        ", ".join(tickers),
        start,
        end,
        interval,
    )

    data = yf.download(tickers, start=start, end=end, interval=interval, progress=False)

    if data.empty:
        raise ValueError(
            f"yfinance returned no data for tickers={tickers}, start={start}, end={end}."
        )

    # Extract price column (handle API differences)
    if isinstance(data.columns, pd.MultiIndex):
        levels = data.columns.get_level_values(0).unique().tolist()
        if "Adj Close" in levels:
            prices = data["Adj Close"]
        elif "Close" in levels:
            prices = data["Close"]
        else:
            raise ValueError(f"No 'Adj Close' or 'Close' in response. Available: {levels}")
    else:
        prices = data

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.astype(float)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    return prices


def validate_and_clean(
    prices: pd.DataFrame,
    drop_threshold: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean price data and report quality statistics.

    Parameters
    ----------
    prices : DataFrame
        Raw price data.
    drop_threshold : float, default 0.1
        If a ticker has more than this fraction of rows as NaN, drop
        the entire column.

    Returns
    -------
    prices : DataFrame
        Cleaned prices (NaN rows dropped after column filtering).
    quality : dict
        Data quality summary per ticker.
    """
    n_rows = len(prices)
    quality: dict[str, int] = {}

    # Check per-column NaN counts
    cols_to_drop = []
    for col in prices.columns:
        n_nan = int(prices[col].isna().sum())
        quality[col] = n_nan
        if n_nan / max(n_rows, 1) > drop_threshold:
            cols_to_drop.append(col)
            logger.warning(
                "Ticker %s has %d/%d NaN values (%.1f%%). Dropping.",
                col,
                n_nan,
                n_rows,
                100 * n_nan / n_rows,
            )

    if cols_to_drop:
        prices = prices.drop(columns=cols_to_drop)

    # Drop rows with any remaining NaN
    n_before = len(prices)
    prices = prices.dropna(how="any")
    n_dropped = n_before - len(prices)

    if n_dropped > 0:
        logger.info(
            "Dropped %d rows with missing data (%.1f%%).",
            n_dropped,
            100 * n_dropped / max(n_before, 1),
        )

    return prices, quality


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical prices from Yahoo Finance.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols (e.g. AAPL MSFT GOOGL).",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        choices=list(UNIVERSES.keys()),
        help="Predefined ticker universe. Ignored if --tickers is set.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Start date, inclusive (default: 2018-01-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-31",
        help="End date, exclusive (default: 2023-12-31).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Sampling interval (default: 1d).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: data/prices_{universe}_{start}_{end}.csv",
    )
    args = parser.parse_args()

    # Resolve tickers
    if args.tickers is not None:
        tickers = args.tickers
        universe_name = "custom"
    elif args.universe is not None:
        tickers = UNIVERSES[args.universe]
        universe_name = args.universe
    else:
        tickers = UNIVERSES["faang"]
        universe_name = "faang"

    # Resolve output path
    if args.output is not None:
        output_path = Path(args.output)
    else:
        safe_start = args.start.replace("-", "")
        safe_end = args.end.replace("-", "")
        output_path = Path(f"data/prices_{universe_name}_{safe_start}_{safe_end}.csv")

    # Fetch
    prices = fetch_prices(
        tickers=tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
    )

    # Clean
    prices, quality = validate_and_clean(prices)

    if prices.empty:
        logger.error("No valid data after cleaning. Exiting.")
        sys.exit(1)

    # Save CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=True)
    logger.info(
        "Saved %d rows x %d tickers to %s",
        len(prices),
        prices.shape[1],
        output_path,
    )

    # Save metadata
    metadata = {
        "tickers": list(prices.columns),
        "universe": universe_name,
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "n_rows": len(prices),
        "n_tickers": prices.shape[1],
        "date_range": {
            "first": str(prices.index[0].date()),
            "last": str(prices.index[-1].date()),
        },
        "nan_counts": quality,
        "fetched_at": datetime.now().isoformat(),
    }

    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved metadata to %s", meta_path)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
