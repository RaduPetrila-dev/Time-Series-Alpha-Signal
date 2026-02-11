"""Risk model for portfolio construction.

Transforms raw signal scores into risk-adjusted portfolio weights.
Each transform is independent and composable via :func:`apply_risk_model`.

Transforms (applied in order):

1. **Signal clipping** -- cap extreme z-scores to reduce concentration.
2. **Signal smoothing** -- exponential decay to reduce turnover.
3. **Inverse-volatility weighting** -- scale positions by 1/vol for
   equal risk contribution.
4. **Volatility targeting** -- scale the portfolio to a target
   annualised volatility.

References
----------
Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
Pedersen, L. H. (2015). *Efficiently Inefficient*, Ch. 8 (Risk Management).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk model transforms.

    Parameters
    ----------
    clip_zscore : float
        Cap absolute signal values at this many standard deviations.
        Set to 0.0 to disable. Default 2.0.
    smooth_halflife : int
        Halflife in days for exponential smoothing of the signal.
        Set to 0 to disable. Default 5.
    inverse_vol : bool
        Scale positions by inverse realised volatility. Default True.
    vol_lookback : int
        Lookback window for volatility estimation. Default 20.
    vol_target : float
        Target annualised portfolio volatility. Set to 0.0 to disable.
        Default 0.15 (15%).
    vol_target_lookback : int
        Lookback for portfolio volatility estimation used in
        vol targeting. Default 63 (quarterly).
    vol_floor : float
        Minimum volatility estimate to avoid division by near-zero.
        Default 0.01 (1% annualised).
    max_leverage : float
        Hard cap on gross leverage after vol targeting. Default 2.0.
    """

    clip_zscore: float = 2.0
    smooth_halflife: int = 5
    inverse_vol: bool = True
    vol_lookback: int = 20
    vol_target: float = 0.15
    vol_target_lookback: int = 63
    vol_floor: float = 0.01
    max_leverage: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise config to a dictionary."""
        return {
            "clip_zscore": self.clip_zscore,
            "smooth_halflife": self.smooth_halflife,
            "inverse_vol": self.inverse_vol,
            "vol_lookback": self.vol_lookback,
            "vol_target": self.vol_target,
            "vol_target_lookback": self.vol_target_lookback,
            "vol_floor": self.vol_floor,
            "max_leverage": self.max_leverage,
        }


def clip_signal(
    signal: pd.DataFrame,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """Clip signal values to [-n_std, +n_std] cross-sectionally."""
    if n_std <= 0:
        return signal
    return signal.clip(lower=-n_std, upper=n_std)


def smooth_signal(
    signal: pd.DataFrame,
    halflife: int = 5,
) -> pd.DataFrame:
    """Apply exponential weighted moving average to reduce turnover."""
    if halflife <= 0:
        return signal
    return signal.ewm(halflife=halflife, min_periods=1).mean()


def inverse_vol_weight(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 20,
    vol_floor: float = 0.01,
) -> pd.DataFrame:
    """Scale each position by inverse realised volatility."""
    returns = prices.pct_change()
    realised_vol = returns.rolling(lookback, min_periods=max(lookback // 2, 2)).std()
    ann_vol = realised_vol * np.sqrt(252)
    ann_vol = ann_vol.clip(lower=vol_floor)
    inv_vol = 1.0 / ann_vol
    weighted = signal * inv_vol
    weighted = weighted.reindex(signal.index).fillna(0.0)
    return weighted


def vol_target_scale(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.15,
    lookback: int = 63,
    vol_floor: float = 0.01,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Scale portfolio weights to achieve a target annualised volatility."""
    if target_vol <= 0:
        return weights
    returns = prices.pct_change()
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    realised_vol = port_ret.rolling(lookback, min_periods=max(lookback // 2, 2)).std()
    ann_port_vol = (realised_vol * np.sqrt(252)).clip(lower=vol_floor)
    scale = (target_vol / ann_port_vol).clip(upper=max_leverage)
    scaled = weights.mul(scale, axis=0)
    return scaled


def normalise_weights(
    weights: pd.DataFrame,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """Normalise portfolio weights to respect gross exposure constraint."""
    row_abs = weights.abs().sum(axis=1).replace(0, np.nan)
    normed = weights.div(row_abs, axis=0).fillna(0.0) * max_gross
    return normed


def apply_risk_model(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    config: RiskConfig | None = None,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """Apply the full risk model pipeline to raw signal scores.

    Pipeline order:
    1. Clip extreme signal values.
    2. Smooth the signal to reduce turnover.
    3. Scale by inverse volatility (equal risk contribution).
    4. Normalise to max_gross.
    5. Apply volatility targeting.
    """
    if config is None:
        config = RiskConfig()

    w = signal.copy()

    if config.clip_zscore > 0:
        w = clip_signal(w, n_std=config.clip_zscore)
        logger.debug("Clipped signal at +/- %.1f std", config.clip_zscore)

    if config.smooth_halflife > 0:
        w = smooth_signal(w, halflife=config.smooth_halflife)
        logger.debug("Smoothed signal with halflife=%d", config.smooth_halflife)

    if config.inverse_vol:
        w = inverse_vol_weight(
            w, prices, lookback=config.vol_lookback, vol_floor=config.vol_floor,
        )
        logger.debug("Applied inverse-vol weighting, lookback=%d", config.vol_lookback)

    w = normalise_weights(w, max_gross=max_gross)

    if config.vol_target > 0:
        w = vol_target_scale(
            w, prices, target_vol=config.vol_target,
            lookback=config.vol_target_lookback,
            vol_floor=config.vol_floor, max_leverage=config.max_leverage,
        )
        logger.debug(
            "Applied vol targeting: target=%.1f%%, max_leverage=%.1f",
            config.vol_target * 100, config.max_leverage,
        )

    return w
