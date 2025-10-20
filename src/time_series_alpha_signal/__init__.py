"""Top‑level package for time_series_alpha_signal.

This module exposes the primary entry points for backtesting and signal
generation.  Import ``backtest`` to run a cross‑sectional strategy, and see
``signals`` for individual signal functions.
"""

from .backtest import backtest  # noqa: F401
from . import signals  # noqa: F401
from .data import load_synthetic_prices, load_csv_prices  # noqa: F401

__all__ = [
    "backtest",
    "signals",
    "load_synthetic_prices",
    "load_csv_prices",
