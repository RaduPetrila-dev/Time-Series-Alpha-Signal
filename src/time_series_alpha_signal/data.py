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
