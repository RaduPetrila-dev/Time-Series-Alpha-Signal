"""Data loading utilities for synthetic, CSV, and Yahoo Finance price data.

This module provides three entry points for obtaining price DataFrames
suitable for the backtesting pipeline:

* :func:`load_synthetic_prices` -- geometric Brownian motion simulator
  with configurable drift and volatility per asset.
* :func:`load_csv_prices` -- read a local CSV with one column per asset.
* :func:`load_yfinance_prices` -- fetch adjusted close prices via the
  ``yfinance`` library (optional dependency).

All loaders return a ``pd.DataFrame`` with a ``DatetimeIndex`` and one
column per asset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: yfinance
# ---------------------------------------------------------------------------

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def load_synthetic_prices(
    n_names: int = 20,
    n_days: int = 750,
    seed: int = 42,
    annual_drift: float = 0.05,
    annual_vol: float = 0.20,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic prices via geometric Brownian motion.

    Each asset follows:

    .. math::
        S_{t+1} = S_t \\exp\\bigl(
            (\\mu - \\tfrac{1}{2}\\sigma^2)\\,\\Delta t
            + \\sigma\\,\\sqrt{\\Delta t}\\,Z_t
        \\bigr)

    where :math:`\\mu` is ``annual_drift``, :math:`\\sigma` is
    ``annual_vol``, :math:`\\Delta t = 1/252`, and
    :math:`Z_t \\sim N(0,1)`.

    Parameters
    ----------
    n_names : int, default 20
        Number of synthetic assets.
    n_days : int, default 750
        Number of trading days.
    seed : int, default 42
        Random seed for reproducibility.
    annual_drift : float, default 0.05
        Annualised expected return (mu).
    annual_vol : float, default 0.20
        Annualised volatility (sigma).
    start_price : float, default 100.0
        Starting price for every asset.

    Returns
    -------
    DataFrame
        Synthetic price series indexed by business dates.

    Raises
    ------
    ValueError
        If *n_names* or *n_days* is less than 1.
    """
    if n_names < 1:
        raise ValueError(f"n_names must be >= 1, got {n_names}")
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(), periods=n_days
    )
    names = [f"SYM{i:03d}" for i in range(n_names)]

    dt = 1.0 / 252.0
    drift = (annual_drift - 0.5 * annual_vol**2) * dt
    diffusion = annual_vol * np.sqrt(dt)

    shocks = rng.normal(0, 1, size=(n_days, n_names))
    log_returns = drift + diffusion * shocks
    prices = start_price * np.exp(np.cumsum(log_returns, axis=0))

    df = pd.DataFrame(prices, index=dates, columns=names).round(4)

    logger.info(
        "Generated synthetic prices: %d assets x %d days "
        "(drift=%.2f, vol=%.2f).",
        n_names,
        n_days,
        annual_drift,
        annual_vol,
    )
    return df


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_csv_prices(
    path: str,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Load price data from a local CSV file.

    The CSV should contain a date column and one numeric column per
    asset.  Non-numeric columns (other than the date column) are
    silently dropped.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    date_col : str, default "Date"
        Name of the column containing dates.

    Returns
    -------
    DataFrame
        Price series indexed by datetime.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If *date_col* is not found in the CSV headers.
    ValueError
        If no numeric price columns remain after filtering.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=[date_col])

    if date_col not in df.columns:
        raise KeyError(
            f"Date column '{date_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.set_index(date_col)
    price_df = df.select_dtypes(include=[np.number]).copy()

    if price_df.empty:
        raise ValueError(
            "No numeric columns found in the CSV after removing "
            f"the date column '{date_col}'."
        )

    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    n_nan = int(price_df.isna().sum().sum())
    if n_nan > 0:
        logger.warning(
            "CSV contains %d NaN values across %d assets. "
            "Consider forward-filling or dropping before backtesting.",
            n_nan,
            price_df.shape[1],
        )

    logger.info(
        "Loaded CSV prices: %d rows x %d assets from %s.",
        len(price_df),
        price_df.shape[1],
        filepath.name,
    )
    return price_df


# ---------------------------------------------------------------------------
# Yahoo Finance loader
# ---------------------------------------------------------------------------


def load_yfinance_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_dropna: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted close prices from Yahoo Finance.

    Uses the ``yfinance`` library to download historical prices for the
    given tickers.  The function handles both single-ticker and
    multi-ticker responses, and adapts to API changes in recent
    ``yfinance`` versions (>= 0.2.31 renamed "Adj Close" to "Close"
    when ``auto_adjust=True``, which is now the default).

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols recognised by Yahoo Finance.
    start : str or None
        Start date inclusive (``YYYY-MM-DD``).  ``None`` downloads the
        maximum available history.
    end : str or None
        End date exclusive (``YYYY-MM-DD``).
    interval : str, default "1d"
        Sampling interval (e.g. ``"1d"``, ``"1wk"``).
    auto_dropna : bool, default True
        Drop rows with any missing values.  Recommended for
        cross-sectional strategies to keep the universe aligned.

    Returns
    -------
    DataFrame
        Price series indexed by datetime with one column per ticker.

    Raises
    ------
    ImportError
        If ``yfinance`` is not installed.
    ValueError
        If *tickers* is empty or no data is returned.
    """
    if yf is None:
        raise ImportError(
            "yfinance is not installed. "
            "Install it with: pip install yfinance"
        )
    if not tickers:
        raise ValueError("tickers list is empty.")

    logger.info(
        "Downloading %d ticker(s) from yfinance: %s",
        len(tickers),
        ", ".join(tickers[:5]) + ("..." if len(tickers) > 5 else ""),
    )

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"yfinance returned no data for tickers={tickers}, "
            f"start={start}, end={end}."
        )

    # Extract the price column from the (possibly multi-level) DataFrame.
    # yfinance >= 0.2.31 with auto_adjust=True (default) returns "Close"
    # as the adjusted column.  Older versions return "Adj Close".
    prices: pd.DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        available_levels = data.columns.get_level_values(0).unique().tolist()
        if "Adj Close" in available_levels:
            prices = data["Adj Close"].copy()
        elif "Close" in available_levels:
            prices = data["Close"].copy()
        else:
            raise ValueError(
                "yfinance response has no 'Adj Close' or 'Close' level. "
                f"Available levels: {available_levels}"
            )
    else:
        # Single ticker returns a flat DataFrame
        prices = data.copy()

    # If a single ticker was requested, ensure the result is a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.astype(float)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    if auto_dropna:
        n_before = len(prices)
        prices = prices.dropna(how="any")
        n_dropped = n_before - len(prices)
        if n_dropped > 0:
            logger.info(
                "Dropped %d rows with missing data (%.1f%%).",
                n_dropped,
                100.0 * n_dropped / max(1, n_before),
            )

    logger.info(
        "yfinance prices: %d rows x %d tickers, "
        "from %s to %s.",
        len(prices),
        prices.shape[1],
        prices.index[0].date() if len(prices) > 0 else "N/A",
        prices.index[-1].date() if len(prices) > 0 else "N/A",
    )

    return prices
