"""Tests for the risk model module."""

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


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", periods=500)
    data = 100 + np.cumsum(rng.standard_normal((500, 5)), axis=0)
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(5)])


@pytest.fixture
def synthetic_signal(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    data = rng.standard_normal(synthetic_prices.shape)
    return pd.DataFrame(
        data, index=synthetic_prices.index, columns=synthetic_prices.columns,
    )


class TestClipSignal:
    def test_values_within_bounds(self, synthetic_signal: pd.DataFrame) -> None:
        clipped = clip_signal(synthetic_signal, n_std=2.0)
        assert clipped.max().max() <= 2.0 + 1e-10
        assert clipped.min().min() >= -2.0 - 1e-10

    def test_disabled_when_zero(self, synthetic_signal: pd.DataFrame) -> None:
        result = clip_signal(synthetic_signal, n_std=0.0)
        pd.testing.assert_frame_equal(result, synthetic_signal)


class TestSmoothSignal:
    def test_reduces_daily_change(self, synthetic_signal: pd.DataFrame) -> None:
        smoothed = smooth_signal(synthetic_signal, halflife=5)
        raw_turnover = synthetic_signal.diff().abs().sum().sum()
        smooth_turnover = smoothed.diff().abs().sum().sum()
        assert smooth_turnover < raw_turnover

    def test_disabled_when_zero(self, synthetic_signal: pd.DataFrame) -> None:
        result = smooth_signal(synthetic_signal, halflife=0)
        pd.testing.assert_frame_equal(result, synthetic_signal)


class TestInverseVolWeight:
    def test_high_vol_gets_lower_weight(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        signal = pd.DataFrame(
            1.0, index=synthetic_prices.index, columns=synthetic_prices.columns,
        )
        weighted = inverse_vol_weight(signal, synthetic_prices, lookback=20)
        vols = (
            synthetic_prices.pct_change()
            .rolling(20, min_periods=10)
            .std()
            .iloc[-1]
        )
        highest_vol_asset = vols.idxmax()
        lowest_vol_asset = vols.idxmin()
        last_row = weighted.iloc[-1]
        assert last_row[lowest_vol_asset] > last_row[highest_vol_asset]


class TestVolTargetScale:
    def test_disabled_when_zero(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = pd.DataFrame(
            0.2, index=synthetic_prices.index, columns=synthetic_prices.columns,
        )
        result = vol_target_scale(weights, synthetic_prices, target_vol=0.0)
        pd.testing.assert_frame_equal(result, weights)


class TestNormaliseWeights:
    def test_gross_exposure(self) -> None:
        w = pd.DataFrame({"A": [1.0, -2.0], "B": [3.0, 0.5]})
        normed = normalise_weights(w, max_gross=1.0)
        gross = normed.abs().sum(axis=1)
        np.testing.assert_allclose(gross.values, [1.0, 1.0], atol=1e-10)


class TestApplyRiskModel:
    def test_default_config(
        self, synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        weights = apply_risk_model(
            synthetic_signal, synthetic_prices, config=RiskConfig(),
        )
        assert weights.shape == synthetic_signal.shape
        assert np.isfinite(weights.values[100:]).all()

    def test_no_transforms(
        self, synthetic_signal: pd.DataFrame,
        synthetic_prices: pd.DataFrame,
    ) -> None:
        config = RiskConfig(
            clip_zscore=0.0, smooth_halflife=0,
            inverse_vol=False, vol_target=0.0,
        )
        weights = apply_risk_model(
            synthetic_signal, synthetic_prices,
            config=config, max_gross=1.0,
        )
        row_abs = synthetic_signal.abs().sum(axis=1).replace(0, np.nan)
        expected = synthetic_signal.div(row_abs, axis=0).fillna(0.0)
        pd.testing.assert_frame_equal(weights, expected, atol=1e-10)


class TestRiskConfig:
    def test_to_dict_keys(self) -> None:
        config = RiskConfig()
        d = config.to_dict()
        expected_keys = {
            "clip_zscore", "smooth_halflife", "inverse_vol",
            "vol_lookback", "vol_target", "vol_target_lookback",
            "vol_floor", "max_leverage",
        }
        assert set(d.keys()) == expected_keys

    def test_frozen(self) -> None:
        config = RiskConfig()
        with pytest.raises(AttributeError):
            config.clip_zscore = 5.0  # type: ignore[misc]
