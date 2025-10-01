import pandas as pd
import numpy as np
from timeseries_alpha.backtest import backtest
from timeseries_alpha.metrics import equity_curve

def test_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    price = pd.Series(np.linspace(100, 150, len(idx)), index=idx, name="A")
    prices = pd.concat([price], axis=1)
    sig = prices.copy()
    res = backtest(prices, sig)
    first_w_date = res["weights"].dropna().index[0]
    assert first_w_date > idx[0]

def test_equity_curve_monotonic_with_positive_returns():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    rets = pd.Series([0.01]*10, index=idx)
    ec = equity_curve(rets)
    assert (ec.diff().dropna() > 0).all()
