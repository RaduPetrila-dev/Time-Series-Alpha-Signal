from __future__ import annotations
import pandas as pd
import numpy as np

def load_synthetic_prices(n_names: int = 20, n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    names = [f"SYM{i:03d}" for i in range(n_names)]
    # geometric random walk
    shocks = rng.normal(0, 0.01, size=(n_days, n_names))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=names).round(4)
    return df
