"""Tests for the meta-labelling model and helpers.

Covers:
* train_meta_model end-to-end with valid and edge-case inputs.
* _compute_realized_returns correctness.
* build_features output shape and no lookahead.
* MetaModelResult structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_series_alpha_signal.models import (
    MetaModelResult,
    _compute_realized_returns,
    build_features,
    train_meta_model,
)
from time_series_alpha_signal.labels import triple_barrier_labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_prices() -> pd.Series:
    """Random walk price series (150 days)."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2021-01-01", periods=150)
    return pd.Series(100 + np.cumsum(rng.standard_normal(150)), index=dates)


@pytest.fixture
def longer_prices() -> pd.Series:
    """Longer random walk (500 days) for more stable CV."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500)
    return pd.Series(100 + np.cumsum(rng.standard_normal(500)), index=dates)


# ---------------------------------------------------------------------------
# train_meta_model
# ---------------------------------------------------------------------------


class TestTrainMetaModel:
    """Tests for the end-to-end meta-model training pipeline."""

    def test_returns_expected_keys(self, random_prices: pd.Series) -> None:
        metrics = train_meta_model(random_prices, lookback=10, n_splits=3)
        expected_keys = {
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "fold_accuracies",
            "n_samples",
            "n_positive",
            "n_negative",
        }
        assert set(metrics.keys()) == expected_keys

    def test_accuracy_in_valid_range(self, random_prices: pd.Series) -> None:
        metrics = train_meta_model(random_prices, lookback=10, n_splits=3)
        if not np.isnan(metrics["cv_accuracy_mean"]):
            assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0
            assert 0.0 <= metrics["cv_accuracy_std"]

    def test_fold_accuracies_length(self, random_prices: pd.Series) -> None:
        n_splits = 3
        metrics = train_meta_model(
            random_prices, lookback=10, n_splits=n_splits
        )
        if not np.isnan(metrics["cv_accuracy_mean"]):
            assert len(metrics["fold_accuracies"]) <= n_splits
            for acc in metrics["fold_accuracies"]:
                assert 0.0 <= acc <= 1.0

    def test_class_balance_reported(self, random_prices: pd.Series) -> None:
        metrics = train_meta_model(random_prices, lookback=10, n_splits=3)
        assert metrics["n_positive"] >= 0
        assert metrics["n_negative"] >= 0
        assert metrics["n_positive"] + metrics["n_negative"] == metrics["n_samples"]

    def test_n_samples_positive(self, longer_prices: pd.Series) -> None:
        metrics = train_meta_model(longer_prices, lookback=10, n_splits=5)
        assert metrics["n_samples"] > 0

    def test_insufficient_data_returns_nan(self) -> None:
        """Very short series should return NaN metrics gracefully."""
        dates = pd.date_range("2021-01-01", periods=10)
        rng = np.random.default_rng(99)
        prices = pd.Series(100 + np.cumsum(rng.standard_normal(10)), index=dates)
        metrics = train_meta_model(prices, lookback=5, n_splits=3)
        assert np.isnan(metrics["cv_accuracy_mean"])

    def test_different_parameters(self, longer_prices: pd.Series) -> None:
        """Model should run with non-default parameters."""
        metrics = train_meta_model(
            longer_prices,
            lookback=5,
            vol_span=30,
            pt_sl=(0.5, 0.5),
            horizon=10,
            n_splits=4,
            embargo_pct=0.02,
        )
        assert "cv_accuracy_mean" in metrics


# ---------------------------------------------------------------------------
# _compute_realized_returns
# ---------------------------------------------------------------------------


class TestComputeRealizedReturns:
    """Tests for the vectorised realised return computation."""

    def test_known_return(self) -> None:
        """Check return = (p1 - p0) / p0 for a known case."""
        dates = pd.date_range("2021-01-01", periods=5)
        prices = pd.Series([100.0, 105.0, 110.0, 108.0, 112.0], index=dates)
        events = pd.DataFrame(
            {"t1": [dates[2], dates[4]], "y": [1, -1]},
            index=[dates[0], dates[1]],
        )
        returns = _compute_realized_returns(prices, events)
        assert len(returns) == 2
        assert abs(returns.iloc[0] - 0.10) < 1e-6  # (110-100)/100
        assert abs(returns.iloc[1] - (112 - 105) / 105) < 1e-6

    def test_output_aligned_to_event_index(
        self, random_prices: pd.Series
    ) -> None:
        events = triple_barrier_labels(random_prices, horizon=5)
        returns = _compute_realized_returns(random_prices, events)
        assert returns.index.equals(events.index)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    """Tests for the feature construction function."""

    def test_output_columns(self, random_prices: pd.Series) -> None:
        features = build_features(random_prices, lookback=10)
        expected_cols = {"momentum", "volatility", "mean_rev"}
        assert set(features.columns) == expected_cols

    def test_output_length(self, random_prices: pd.Series) -> None:
        features = build_features(random_prices, lookback=10)
        assert len(features) == len(random_prices)

    def test_no_lookahead(self, random_prices: pd.Series) -> None:
        """Features at time t should not depend on prices after t.

        Verify by checking that truncating the last 10 rows does not
        change features computed on the earlier rows.
        """
        full_feat = build_features(random_prices, lookback=10)
        truncated = random_prices.iloc[:-10]
        trunc_feat = build_features(truncated, lookback=10)

        # Features on the truncated series should match the
        # corresponding rows of the full series
        overlap = trunc_feat.index
        pd.testing.assert_frame_equal(
            full_feat.loc[overlap],
            trunc_feat,
            check_names=False,
        )

    def test_invalid_lookback_raises(self, random_prices: pd.Series) -> None:
        with pytest.raises(ValueError, match="lookback"):
            build_features(random_prices, lookback=0)


# ---------------------------------------------------------------------------
# MetaModelResult dataclass
# ---------------------------------------------------------------------------


class TestMetaModelResult:
    """Tests for the result container."""

    def test_to_dict_roundtrip(self) -> None:
        result = MetaModelResult(
            cv_accuracy_mean=0.65,
            cv_accuracy_std=0.03,
            fold_accuracies=[0.62, 0.68, 0.65],
            n_samples=100,
            n_positive=55,
            n_negative=45,
        )
        d = result.to_dict()
        assert d["cv_accuracy_mean"] == 0.65
        assert d["n_samples"] == 100
        assert len(d["fold_accuracies"]) == 3

    def test_frozen(self) -> None:
        result = MetaModelResult(
            cv_accuracy_mean=0.5,
            cv_accuracy_std=0.0,
            fold_accuracies=[],
            n_samples=0,
            n_positive=0,
            n_negative=0,
        )
        with pytest.raises(AttributeError):
            result.cv_accuracy_mean = 0.9  # type: ignore[misc]
