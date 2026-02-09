"""Tests for the signal combination engine.

Covers:
* Z-score normalisation correctness.
* Equal-weight blend produces valid metrics.
* Walk-forward optimisation runs end-to-end.
* Weight grid generation.
* Edge cases (insufficient data, single signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from time_series_alpha_signal.combiner import (
    CombinedResult,
    WalkForwardResult,
    _generate_weight_grid,
    blend_equal_weight,
    compute_signals,
    walk_forward_optimize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Random walk prices, 5 assets, 800 days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=800)
    data = 100 + np.cumsum(rng.standard_normal((800, 5)), axis=0)
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(5)])


@pytest.fixture
def short_prices() -> pd.DataFrame:
    """Short series for edge case testing."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-01", periods=100)
    data = 100 + np.cumsum(rng.standard_normal((100, 3)), axis=0)
    return pd.DataFrame(data, index=dates, columns=["X", "Y", "Z"])


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------


class TestZScore:
    def test_mean_zero(self, synthetic_prices: pd.DataFrame) -> None:
        signals = compute_signals(synthetic_prices, ["momentum"])
        z = signals["momentum"]
        # Each row should have ~zero mean after z-scoring
        row_means = z.dropna().mean(axis=1)
        assert row_means.abs().max() < 1e-10

    def test_unit_std(self, synthetic_prices: pd.DataFrame) -> None:
        signals = compute_signals(synthetic_prices, ["momentum"])
        z = signals["momentum"]
        row_stds = z.dropna().std(axis=1)
        # Std should be ~1 for rows with variance
        valid = row_stds[row_stds > 0]
        assert (valid - 1.0).abs().max() < 1e-10


# ---------------------------------------------------------------------------
# Weight grid
# ---------------------------------------------------------------------------


class TestWeightGrid:
    def test_weights_sum_to_one(self) -> None:
        grid = _generate_weight_grid(3, step=0.2)
        for combo in grid:
            assert abs(sum(combo) - 1.0) < 1e-6

    def test_grid_not_empty(self) -> None:
        grid = _generate_weight_grid(2, step=0.2)
        assert len(grid) > 0

    def test_two_signals_grid_size(self) -> None:
        grid = _generate_weight_grid(2, step=0.2)
        # (0,1), (0.2,0.8), (0.4,0.6), (0.6,0.4), (0.8,0.2), (1,0)
        assert len(grid) == 6


# ---------------------------------------------------------------------------
# Equal-weight blend
# ---------------------------------------------------------------------------


class TestEqualWeightBlend:
    def test_returns_combined_result(self, synthetic_prices: pd.DataFrame) -> None:
        result = blend_equal_weight(
            synthetic_prices,
            signal_names=["momentum", "mean_reversion"],
            cost_bps=5.0,
        )
        assert isinstance(result, CombinedResult)
        assert result.mode == "equal_weight"

    def test_metrics_keys(self, synthetic_prices: pd.DataFrame) -> None:
        result = blend_equal_weight(
            synthetic_prices,
            signal_names=["momentum", "ewma_momentum"],
        )
        required_keys = {"days", "cagr", "ann_vol", "sharpe", "max_dd", "signals", "mode"}
        assert required_keys.issubset(result.metrics.keys())

    def test_weights_equal(self, synthetic_prices: pd.DataFrame) -> None:
        signals = ["momentum", "mean_reversion", "ewma_momentum"]
        result = blend_equal_weight(synthetic_prices, signal_names=signals)
        for name in signals:
            assert abs(result.weights_used[name] - 1.0 / 3) < 1e-10

    def test_sharpe_is_finite(self, synthetic_prices: pd.DataFrame) -> None:
        result = blend_equal_weight(
            synthetic_prices,
            signal_names=["momentum", "volatility"],
        )
        assert np.isfinite(result.metrics["sharpe"])

    def test_single_signal_raises(self, synthetic_prices: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            blend_equal_weight(synthetic_prices, signal_names=["momentum"])

    def test_three_signals(self, synthetic_prices: pd.DataFrame) -> None:
        result = blend_equal_weight(
            synthetic_prices,
            signal_names=["momentum", "mean_reversion", "ewma_momentum"],
        )
        assert result.metrics["days"] > 0
        assert len(result.equity) > 0


# ---------------------------------------------------------------------------
# Walk-forward optimisation
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_returns_walk_forward_result(self, synthetic_prices: pd.DataFrame) -> None:
        result = walk_forward_optimize(
            synthetic_prices,
            signal_names=["momentum", "mean_reversion"],
            train_days=300,
            test_days=200,
            step_days=200,
            weight_step=0.5,
        )
        assert isinstance(result, WalkForwardResult)

    def test_oos_metrics_present(self, synthetic_prices: pd.DataFrame) -> None:
        result = walk_forward_optimize(
            synthetic_prices,
            signal_names=["momentum", "ewma_momentum"],
            train_days=300,
            test_days=200,
            step_days=200,
            weight_step=0.5,
        )
        assert "oos_sharpe" in result.overall_metrics
        assert "oos_cagr" in result.overall_metrics
        assert result.overall_metrics["n_windows"] >= 1

    def test_window_results_populated(self, synthetic_prices: pd.DataFrame) -> None:
        result = walk_forward_optimize(
            synthetic_prices,
            signal_names=["momentum", "mean_reversion"],
            train_days=300,
            test_days=200,
            step_days=200,
            weight_step=0.5,
        )
        assert len(result.window_results) >= 1
        w = result.window_results[0]
        assert "best_weights" in w
        assert "train_sharpe" in w
        assert "oos_sharpe" in w

    def test_insufficient_data_raises(self, short_prices: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least"):
            walk_forward_optimize(
                short_prices,
                signal_names=["momentum", "mean_reversion"],
                train_days=504,
                test_days=252,
            )

    def test_oos_equity_monotonic_index(self, synthetic_prices: pd.DataFrame) -> None:
        result = walk_forward_optimize(
            synthetic_prices,
            signal_names=["momentum", "mean_reversion"],
            train_days=300,
            test_days=200,
            step_days=200,
            weight_step=0.5,
        )
        assert result.equity.index.is_monotonic_increasing
