import numpy as np
import pandas as pd

from timeseries_alpha import backtest, prepare_weights_from_signal


def _toy():
    idx = pd.bdate_range("2020-01-01", periods=30)
    cols = list("ABCDE")
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(idx), len(cols))), index=idx, columns=cols)
    sig = rets.rolling(5, min_periods=3).mean()
    return rets, sig


def test_l1_exposure():
    rets, sig = _toy()
    w = prepare_weights_from_signal(sig.shift(1), max_gross=1.23)
    l1 = w.abs().sum(axis=1)
    # Rows with any nonzero get scaled to max_gross; rows with all zeros stay at 0
    assert (l1[(l1 > 0) | (sig.shift(1).abs().sum(axis=1) > 0)] - 1.23).abs().max() < 1e-9


def test_no_lookahead():
    rets, sig = _toy()
    res = backtest(rets, sig, max_gross=1.0)
    # first day should be zero weights because signal is lagged
    assert (res["weights"].iloc[0].abs() < 1e-12).all()


def test_costs_non_negative():
    rets, sig = _toy()
    res = backtest(rets, sig, max_gross=1.0, cost_per_dollar=0.001)
    assert (res["cost"] >= 0).all()
