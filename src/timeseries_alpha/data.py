"""
data.py — robust data loading utilities
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

@dataclass(frozen=True)
class PriceSource:
    name: str = "yahoo"
    cache: bool = True

def _cache_path(ticker: str) -> str:
    safe = "".join(c for c in ticker if c.isalnum() or c in ("-", "_")).upper()
    return os.path.join(DATA_DIR, f"{safe}.csv")

def load_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    source: PriceSource = PriceSource(),
    field: str = "Adj Close",
) -> pd.DataFrame:
    """
    Load daily OHLCV data and return a price matrix (columns = tickers).
    Caches individual ticker CSVs to avoid repeated downloads.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    uniq: List[str] = list(dict.fromkeys(t.upper().strip() for t in tickers if t.strip()))
    frames: List[pd.Series] = []

    for t in uniq:
        ser: Optional[pd.Series] = None

        # Try cache first
        cp = _cache_path(t)
        if source.cache and os.path.exists(cp):
            df = pd.read_csv(cp, parse_dates=["Date"]).set_index("Date")
            if field in df.columns:
                ser = df[field].rename(t)

        # Fallback to download
        if ser is None:
            if yf is None:
                raise RuntimeError("yfinance not available and no cache found for %s" % t)
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            # Align naming with yfinance multiindex or single index
            if isinstance(df.columns, pd.MultiIndex):
                if field in df.columns.get_level_values(0):
                    ser = df[field].rename(t)  # type: ignore
            else:
                if field in df.columns:
                    ser = df[field].rename(t)
            if source.cache:
                # Ensure Date index for cache
                to_cache = df.copy()
                to_cache.index.name = "Date"
                to_cache.to_csv(cp)

        if ser is not None and not ser.empty:
            frames.append(ser)

    if not frames:
        raise ValueError("No price data returned. Check tickers/date range/connection.")

    prices = pd.concat(frames, axis=1).sort_index()
    # Forward-fill and drop all-nan rows
    prices = prices.ffill().dropna(how="all")
    return prices

def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns; safe against inf/nans."""
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    return rets
