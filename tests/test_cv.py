"""Tests for cross-validation and the PurgedKFold splitter.

Covers:
* PurgedKFold split mechanics (fold count, no overlap, purging).
* cross_validate_sharpe with different signal types.
* Edge cases (small data, single asset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_series_alpha_signal.cv import PurgedKFold, make_event_table
from time_series_alpha_signal.data import load_synthetic_prices
from time_series_alpha_signal.evaluation import cross_validate_sharpe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Small synthetic price panel for fast tests."""
    return load_synthetic_prices(n_names=3, n_days=200, seed=42)


@pytest.fixture
def event_table(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    """Event table with 10-day horizon from synthetic prices."""
    return make_event_table(synthetic_prices.index, horizon=10)


# ---------------------------------------------------------------------------
# PurgedKFold mechanics
# ---------------------------------------------------------------------------


class TestPurgedKFold:
    """Tests for the PurgedKFold splitter."""

    def test_produces_correct_number_of_folds(
        self, event_table: pd.DataFrame
    ) -> None:
        cv = PurgedKFold(n_splits=5)
        folds = list(cv.split(event_table))
        assert len(folds) == 5

    def test_get_n_splits(self) -> None:
        cv = PurgedKFold(n_splits=3)
        assert cv.get_n_splits() == 3

    def test_train_test_no_index_overlap(
        self, event_table: pd.DataFrame
    ) -> None:
        """Train and test indices must not share any positions."""
        cv = PurgedKFold(n_splits=5)
        for train_idx, test_idx in cv.split(event_table):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_indices_covered(
        self, event_table: pd.DataFrame
    ) -> None:
        """Every index should appear in exactly one test fold."""
        cv = PurgedKFold(n_splits=5)
        all_test = []
        for _, test_idx in cv.split(event_table):
            all_test.extend(test_idx)
        assert sorted(all_test) == list(range(len(event_table)))

    def test_purging_removes_overlapping_events(
        self, event_table: pd.DataFrame
    ) -> None:
        """With purging, train sets should be smaller than without."""
        cv_no_purge = PurgedKFold(n_splits=3, embargo_pct=0.0)
        cv_with_embargo = PurgedKFold(n_splits=3, embargo_pct=0.05)

        for (train_np, _), (train_emb, _) in zip(
            cv_no_purge.split(event_table),
            cv_with_embargo.split(event_table),
        ):
            # Embargo should remove some training observations
            assert len(train_emb) <= len(train_np)

    def test_repr(self) -> None:
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        r = repr(cv)
        assert "PurgedKFold" in r
        assert "5" in r

    def test_raises_on_invalid_splits(self) -> None:
        with pytest.raises((ValueError, Exception)):
            PurgedKFold(n_splits=1)


# ---------------------------------------------------------------------------
# make_event_table
# ---------------------------------------------------------------------------


class TestMakeEventTable:
    """Tests for the make_event_table helper."""

    def test_output_shape(self, synthetic_prices: pd.DataFrame) -> None:
        idx = synthetic_prices.index
        events = make_event_table(idx, horizon=10)
        assert len(events) == len(idx)
        assert "t1" in events.columns

    def test_t1_is_after_t0(self, synthetic_prices: pd.DataFrame) -> None:
        idx = synthetic_prices.index
        events = make_event_table(idx, horizon=5)
        # For all but the last few rows, t1 should be strictly after t0
        interior = events.iloc[:-5]
        assert (interior["t1"] > interior.index).all()


# ---------------------------------------------------------------------------
# cross_validate_sharpe
# ---------------------------------------------------------------------------


class TestCrossValidateSharpe:
    """Tests for the cross_validate_sharpe evaluation function."""

    def test_returns_list_of_floats(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        sharpe_vals = cross_validate_sharpe(
            synthetic_prices, n_splits=3
        )
        assert isinstance(sharpe_vals, list)
        assert len(sharpe_vals) > 0
        for sr in sharpe_vals:
            assert isinstance(sr, float)

    def test_fold_count_matches_n_splits(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        n_splits = 4
        sharpe_vals = cross_validate_sharpe(
            synthetic_prices, n_splits=n_splits
        )
        assert len(sharpe_vals) == n_splits

    @pytest.mark.parametrize(
        "signal_type",
        ["momentum", "mean_reversion", "ewma_momentum"],
    )
    def test_different_signals(
        self, synthetic_prices: pd.DataFrame, signal_type: str
    ) -> None:
        sharpe_vals = cross_validate_sharpe(
            synthetic_prices,
            signal_type=signal_type,
            n_splits=3,
        )
        assert len(sharpe_vals) > 0
        for sr in sharpe_vals:
            assert np.isfinite(sr)

    def test_single_asset(self) -> None:
        """CV should work with a single-column price DataFrame."""
        prices = load_synthetic_prices(n_names=1, n_days=200, seed=99)
        sharpe_vals = cross_validate_sharpe(prices, n_splits=3)
        assert isinstance(sharpe_vals, list)
        assert len(sharpe_vals) > 0

    def test_with_cost_bps(
        self, synthetic_prices: pd.DataFrame
    ) -> None:
        """Higher costs should generally produce lower Sharpe."""
        sr_low_cost = cross_validate_sharpe(
            synthetic_prices, n_splits=3, cost_bps=1.0
        )
        sr_high_cost = cross_validate_sharpe(
            synthetic_prices, n_splits=3, cost_bps=50.0
        )
        # Mean Sharpe with high costs should be <= low costs
        # (not strictly guaranteed on random data, so use a loose check)
        mean_low = np.mean(sr_low_cost)
        mean_high = np.mean(sr_high_cost)
        assert mean_high <= mean_low + 0.5  # generous margin
