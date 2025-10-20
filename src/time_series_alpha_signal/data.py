from __future__ import annotations
import pandas as pd
import numpy as np

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
