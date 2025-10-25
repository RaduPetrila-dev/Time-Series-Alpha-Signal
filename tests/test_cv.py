from time_series_alpha_signal import cross_validate_sharpe, load_synthetic_prices


def test_cross_validate_sharpe_basic() -> None:
    """Cross‑validation returns a non‑empty list of floats."""
    # generate a simple synthetic price panel
    prices = load_synthetic_prices(n_names=3, n_days=100, seed=123)
    sharpe_vals = cross_validate_sharpe(prices, n_splits=3)
    assert isinstance(sharpe_vals, list)
    # At least one fold should be produced
    assert len(sharpe_vals) > 0
    for sr in sharpe_vals:
        # Each entry should be a float (or convertible to float)
        assert isinstance(sr, float)
