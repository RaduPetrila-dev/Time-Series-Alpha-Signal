from __future__ import annotations
import pandas as pd
import numpy as np
try:
    # yfinance is an optional dependency used for pulling real market data.  It
    # is only imported when load_yfinance_prices is called.  This approach
    # avoids a hard runtime dependency for users who only run synthetic or CSV
    # based backtests.  If yfinance is not installed an ImportError will be
    # raised when attempting to call load_yfinance_prices.
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # yfinance is optional

def load_synthetic_prices(n_names: int = 20, n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic price data using a geometric random walk.

    Parameters
    ----------
    n_names : int, default 20
        Number of synthetic symbols to generate.
    n_days : int, default 750
        Number of trading days of data to generate.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Synthetic price series indexed by business dates with symbol columns.
    """
    rng = np.random.default_rng(seed)
    # generate business day index ending at today
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    names = [f"SYM{i:03d}" for i in range(n_names)]
    # simulate log returns and convert to price levels
    shocks = rng.normal(0, 0.01, size=(n_days, n_names))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=names).round(4)
    return df


def load_csv_prices(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load price data from a CSV file.

    The CSV should contain a column with dates and one column per asset.  The
    date column is parsed and used as the index.  All other columns are
    treated as price series.  The function does not perform any resampling
    or forward filling – users should ensure the data are properly aligned
    and cleaned prior to backtesting.

    Parameters
    ----------
    path : str
        Path to the CSV file containing price data.  The file must be
        accessible on the local filesystem.
    date_col : str, default "Date"
        Name of the column containing dates.  If the file uses another
        heading (e.g. "Datetime"), pass that column name here.

    Returns
    -------
    DataFrame
        Price series indexed by dates with asset columns.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col)
    # ensure numeric columns only
    price_df = df.select_dtypes(include=[float, int]).copy()
    price_df.index = pd.to_datetime(price_df.index)
    return price_df


def load_yfinance_prices(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
    auto_dropna: bool = True,
) -> pd.DataFrame:
    """Fetch historical price data from Yahoo Finance for the given tickers.

    This helper uses the ``yfinance`` library to download adjusted close prices
    for a list of symbols.  The data are returned as a wide DataFrame with a
    DatetimeIndex and one column per ticker.  Missing rows or columns can be
    dropped by setting ``auto_dropna=True``.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to download.  Symbols should be recognised by
        Yahoo Finance (e.g. "AAPL", "GOOGL").
    start : str
        Start date (inclusive) in ``YYYY-MM-DD`` format.
    end : str
        End date (exclusive) in ``YYYY-MM-DD`` format.
    interval : str, default "1d"
        Sampling interval for the price series (e.g. "1d", "1wk").  See
        ``yfinance.download`` documentation for supported values.
    auto_dropna : bool, default True
        Whether to drop rows with any missing values.  For cross‑sectional
        strategies it is generally desirable to remove dates where some
        assets are missing to maintain alignment across the universe.

    Returns
    -------
    DataFrame
        Price series indexed by dates with ticker columns.

    Raises
    ------
    ImportError
        If ``yfinance`` is not installed.  To install it run
        ``pip install yfinance``.
    """
    if yf is None:
        raise ImportError(
            "yfinance is not installed. Please install it to load real market data "
            "(e.g. pip install yfinance)."
        )
    # download adjusted close prices; progress=False suppresses verbose output
    data = yf.download(
        tickers, start=start, end=end, interval=interval, progress=False
    )
    # yfinance returns a dict-like structure when multiple tickers are
    # downloaded.  Extract the adjusted close prices consistently.  For a
    # single ticker, ``data`` is a Series; convert to DataFrame for uniformity.
    if hasattr(data, "columns"):
        # Multi-level columns: (attribute, ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # select the Adj Close level
            prices = data["Adj Close"].copy()
        else:
            prices = data.copy()
    else:
        # fallback: convert Series to DataFrame
        prices = data.to_frame(name=tickers[0])
    # ensure all data are floats
    prices = prices.astype(float)
    prices.index = pd.to_datetime(prices.index)
    if auto_dropna:
        prices = prices.dropna(how="any")
    return prices
