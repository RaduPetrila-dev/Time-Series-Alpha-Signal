"""Tests for the risk model module.

Covers:
* Signal clipping bounds.
* Signal smoothing reduces turnover.
* Inverse-vol weighting scales by 1/vol.
* Vol targeting adjusts leverage.
* Full pipeline integration.
* RiskConfig serialisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_series_alpha_signal.risk_model import (
    RiskConfig,
    apply_risk_model,
    clip_signal,
    inverse_vol_weight,
    normalise_weights,
    smooth_signal,
    vol_target_scale,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Random walk prices, 5 assets, 500 days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", periods=500)
    data = 100 + np.cumsum(rng.standard_normal((500, 5)), axis=0)
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(5)])


@pytest.fixture
def synthetic_signal(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Z-scored signal matching synthetic_prices shape."""
    rng = np.random.default_rng(99)
    data = rng.standard_normal(synthetic_prices.shape)
    return pd.DataFrame(
        data,
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------


class TestClipSignal:
    def test_values_within_bounds(self, synthetic_signal: pd.DataFrame) -> None:
        clipped = clip_signal(synthetic_signal, n_std=2.0)
        assert clipped.max().max() <= 2.0 + 1e-10
        assert clipped.min().min() >= -2.0 - 1e-10

    def test_disabled_when_zero(self, synthetic_signal: pd.DataFrame) -> None:
        result = clip_signal(synthetic_signal, n_std=0.0)
        pd.testing.assert_frame_equal(result, synthetic_signal)

    def test_tight_clip(self, synthetic_signal: pd.DataFrame) -> None:
        clipped = clip_signal(synthetic_signal, n_std=0.5)
        assert clipped.max().max() <= 0.5 + 1e-10


# ---------------------------------------------------------------------------
# Smooth
# ---------------------------------------------------------------------------


class TestSmoothSignal:
    def test_reduces_daily_change(self, synthetic_signal: pd.DataFrame) -> None:
        smoothed = smooth_signal(synthetic_signal, halflife=5)
        raw_turnover = synthetic_signal.diff().abs().sum().sum()
        smooth_turnover = smoothed.diff().abs().sum().sum()
        assert smooth_turnover < raw_turnover

    def test_disabled_when_zero(self, synthetic_signal: pd.DataFrame) -> None:
        result = smooth_signal(synthetic_signal, halflife=0)
        pd.testing.assert_frame_equal(result, synthetic_signal)

    def test_shape_preserved(self, synthetic_signal: pd.DataFrame) -> None:
        smoothed = smooth_signal(synthetic_signal, halflife=10)
        assert smoothed.shape == synthetic_signal.shape


# ---------------------------------------------------------------------------
# Inverse-vol
# ---------------------------------------------------------------------------


class TestInverseVolWeight:
    def test_high_vol_gets_lower_weight(
        self,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        # Create signal where all assets have equal score
        signal = pd.DataFrame(
            1.0,
            index=synthetic_prices.index,
            columns=synthetic_prices.columns,
        )
        weighted = inverse_vol_weight(signal, synthetic_prices, lookback=20)

        # Compute vols
        vols = synthetic_prices.pct_change().rolling(20, min_periods=10).std().iloc[-1]
        highest_vol_asset = vols.idxmax()
        lowest_vol_asset = vols.idxmin()

        last_row = weighted.iloc[-1]
        assert last_row[lowest_vol_asset] > last_row[highest_vol_asset]

    def test_vol_floor_prevents_inf(
        self,
        synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        weighted = inverse_vol_weight(
            synthetic_signal,
            synthetic_prices,
            vol_floor=0.01,
        )
        assert np.isfinite(weighted.values).all()


# ---------------------------------------------------------------------------
# Vol targeting
# ---------------------------------------------------------------------------


class TestVolTargetScale:
    def test_output_shape(
        self,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = pd.DataFrame(
            0.2,
            index=synthetic_prices.index,
            columns=synthetic_prices.columns,
        )
        scaled = vol_target_scale(weights, synthetic_prices, target_vol=0.10)
        assert scaled.shape == weights.shape

    def test_disabled_when_zero(
        self,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = pd.DataFrame(
            0.2,
            index=synthetic_prices.index,
            columns=synthetic_prices.columns,
        )
        result = vol_target_scale(weights, synthetic_prices, target_vol=0.0)
        pd.testing.assert_frame_equal(result, weights)

    def test_leverage_cap(
        self,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = pd.DataFrame(
            0.2,
            index=synthetic_prices.index,
            columns=synthetic_prices.columns,
        )
        scaled = vol_target_scale(
            weights,
            synthetic_prices,
            target_vol=0.50,
            max_leverage=1.5,
        )
        gross = scaled.abs().sum(axis=1)
        input_gross = weights.abs().sum(axis=1)
        # Leverage should never exceed max_leverage * input_gross
        ratio = gross / input_gross
        assert ratio.max() <= 1.5 + 1e-6


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------


class TestNormaliseWeights:
    def test_gross_exposure(self) -> None:
        w = pd.DataFrame({"A": [1.0, -2.0], "B": [3.0, 0.5]})
        normed = normalise_weights(w, max_gross=1.0)
        gross = normed.abs().sum(axis=1)
        np.testing.assert_allclose(gross.values, [1.0, 1.0], atol=1e-10)

    def test_zero_row(self) -> None:
        w = pd.DataFrame({"A": [0.0, 1.0], "B": [0.0, -1.0]})
        normed = normalise_weights(w, max_gross=1.0)
        assert normed.iloc[0].abs().sum() == 0.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestApplyRiskModel:
    def test_default_config(
        self,
        synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = apply_risk_model(
            synthetic_signal,
            synthetic_prices,
            config=RiskConfig(),
        )
        assert weights.shape == synthetic_signal.shape
        assert np.isfinite(weights.values[100:]).all()

    def test_no_transforms(
        self,
        synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        config = RiskConfig(
            clip_zscore=0.0,
            smooth_halflife=0,
            inverse_vol=False,
            vol_target=0.0,
        )
        weights = apply_risk_model(
            synthetic_signal,
            synthetic_prices,
            config=config,
            max_gross=1.0,
        )
        # Should be equivalent to simple normalisation
        row_abs = synthetic_signal.abs().sum(axis=1).replace(0, np.nan)
        expected = synthetic_signal.div(row_abs, axis=0).fillna(0.0)
        pd.testing.assert_frame_equal(weights, expected, atol=1e-10)

    def test_none_config_gives_defaults(
        self,
        synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        w1 = apply_risk_model(
            synthetic_signal,
            synthetic_prices,
            config=None,
        )
        w2 = apply_risk_model(
            synthetic_signal,
            synthetic_prices,
            config=RiskConfig(),
        )
        pd.testing.assert_frame_equal(w1, w2)


# ---------------------------------------------------------------------------
# RiskConfig
# ---------------------------------------------------------------------------


class TestRiskConfig:
    def test_to_dict_keys(self) -> None:
        config = RiskConfig()
        d = config.to_dict()
        expected_keys = {
            "clip_zscore",
            "smooth_halflife",
            "inverse_vol",
            "vol_lookback",
            "vol_target",
            "vol_target_lookback",
            "vol_floor",
            "max_leverage",
        }
        assert set(d.keys()) == expected_keys

    def test_frozen(self) -> None:
        config = RiskConfig()
        with pytest.raises(AttributeError):
            config.clip_zscore = 5.0  # type: ignore[misc]
