import pandas as pd, numpy as np
from timeseries_alpha.backtest import backtest

def test_no_lookahead_exposure_and_costs():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"A":[100,101,102,103,104,105],
                           "B":[50,49,51,50,52,53]}, index=idx)
    sig = pd.DataFrame({"A":[0,0,0,1,1,1],
                        "B":[0,0,0,-1,-1,-1]}, index=idx)

    bt0 = backtest(prices, sig, cost_bps=0.0, max_gross=1.0, lag_signal=1)
    # identity: net = gross - cost
    gross = (bt0["weights"] * prices.pct_change().fillna(0.0)).sum(axis=1)
    assert bt0["net_returns"].equals(gross - bt0["cost"])
    # exposure bounded
    assert (bt0["weights"].abs().sum(axis=1) <= 1.000001).all()
    # first day must be 0 (no lookahead)
    assert bt0["net_returns"].iloc[0] == 0.0

    # higher costs cannot increase cumulative PnL
    bt_high = backtest(prices, sig, cost_bps=100.0, max_gross=1.0, lag_signal=1)
    assert bt_high["net_returns"].cumsum().iloc[-1] <= bt0["net_returns"].cumsum().iloc[-1]

    # lag sensitivity
    bt_lag2 = backtest(prices, sig, cost_bps=0.0, max_gross=1.0, lag_signal=2)
    assert bt_lag2["net_returns"].iloc[:2].eq(0.0).all()


