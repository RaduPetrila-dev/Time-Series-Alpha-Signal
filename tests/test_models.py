import numpy as np
import pandas as pd

from time_series_alpha_signal.models import train_meta_model


def test_train_meta_model_basic() -> None:
    """Meta‑labelling model returns metrics on a random series."""
    # random walk prices
    rng = np.random.default_rng(123)
    dates = pd.date_range("2021-01-01", periods=150)
    prices = pd.Series(100 + np.cumsum(rng.standard_normal(150)), index=dates)
    metrics = train_meta_model(prices, lookback=10, n_splits=3)
    # expected keys
    assert set(metrics.keys()) == {"cv_accuracy_mean", "cv_accuracy_std", "n_samples"}
    # if we have enough samples, accuracy mean should be between 0 and 1
    if not np.isnan(metrics["cv_accuracy_mean"]):
        assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0
