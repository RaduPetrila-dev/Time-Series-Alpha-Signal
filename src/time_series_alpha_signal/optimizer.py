"""
Portfolio optimisation utilities.

The functions in this module apply simple post‑processing to portfolio weights
to enforce constraints such as maximum gross leverage.  More complex
optimisation routines (e.g. risk parity, mean–variance) could be.

"""

from __future__ import annotations

import pandas as pd


def enforce_leverage(weights: pd.DataFrame, max_leverage: float = 1.0) -> pd.DataFrame:
    """Scale weights to ensure the L1 norm does not exceed ``max_leverage``.

    For each date the gross leverage is computed as the sum of absolute
    weights.  If this exceeds the allowed ``max_leverage`` the entire
    weight vector is scaled down proportionally.  Otherwise weights are
    returned unchanged.

    Parameters
    ----------
    weights : DataFrame
        Portfolio weights with datetime index and asset columns.
    max_leverage : float, default 1.0
        Maximum allowable gross exposure per day (Σ|weights|).

    Returns
    -------
    DataFrame
        Scaled weights respecting the leverage constraint.
    """
    gross = weights.abs().sum(axis=1)
    scale = (gross / max_leverage).clip(lower=1.0)
    return weights.div(scale, axis=0)
