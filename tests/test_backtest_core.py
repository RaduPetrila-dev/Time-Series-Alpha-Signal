import pandas as pd, numpy as np
from timeseries_alpha.backtest import backtest

def test_no_lookahead_and_exposure_and_costs():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {"A":[100,101,102,103,104,105], "B":[50,49,51,50,52,53]},
        index=idx
    )
    # Simple deterministic signal; weights must be lagged by 1 day in backtest
    sig = pd.DataFrame(
        {"A":[0,0,0,1,1,1], "B":[0,0,0,-1,-1,-1]},
        index=idx
    )

    # Base run
    bt0 = backtest(prices, sig, cost_bps=0.0, max_gross=1.0, lag_signal=1)
    w = bt0["weights"]
    gross = (w * prices.pct_change().fillna(0.0)).sum(axis=1)
    # Identity: net = gross - cost
    assert bt0["net_returns"].equals(gross - bt0["cost"])

    # Exposure bounded
    assert (w.abs().sum(axis=1) <= 1.000001).all()

    # No-lookahead: first day net return must be 0 (no prior info)
    assert bt0["net_returns"].iloc[0] == 0.0

    # Higher costs must not increase cumulative PnL
    bt_high = backtest(prices, sig, cost_bps=100.0, max_gross=1.0, lag_signal=1)
    assert bt_high["net_returns"].cumsum().iloc[-1] <= bt0["net_returns"].cumsum().iloc[-1]

    # Lag sensitivity: with lag=2, first two days must be 0
    bt_lag2 = backtest(prices, sig, cost_bps=0.0, max_gross=1.0, lag_signal=2)
    assert bt_lag2["net_returns"].iloc[:2].eq(0.0).all()



