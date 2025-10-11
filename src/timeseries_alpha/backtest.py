from __future__ import annotations
import pandas as pd
import numpy as np

def momentum_signal(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    rets = prices.pct_change()
    mom = rets.rolling(lookback).sum()
    return mom.shift(1)  # strict lag, no lookahead

def normalize_weights(scores: pd.DataFrame, max_gross: float = 1.0) -> pd.DataFrame:
    # cross-sectional rank to weights, long-short, L1 exposure control
    ranks = scores.rank(axis=1, method="first")
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    raw = centered.div(centered.abs().sum(axis=1), axis=0).fillna(0.0)
    return raw * max_gross

def apply_costs(returns: pd.DataFrame, weights: pd.DataFrame, bps: float = 10.0) -> pd.Series:
    # turnover between t-1 and t weights
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    cost_per_day = turnover * (bps / 1e4)
    gross = (weights.shift(1) * returns).sum(axis=1)
    net = gross - cost_per_day
    return net

def backtest(prices: pd.DataFrame, lookback: int = 20, max_gross: float = 1.0, cost_bps: float = 10.0, seed: int = 42) -> dict:
    prices = prices.sort_index()
    scores = momentum_signal(prices, lookback=lookback)
    weights = normalize_weights(scores, max_gross=max_gross)
    rets = prices.pct_change().fillna(0.0)
    daily = apply_costs(rets, weights, bps=cost_bps)
    equity = (1 + daily).cumprod()
    metrics = {
        "days": int(daily.shape[0]),
        "cagr": float((equity.iloc[-1] ** (252 / max(1, daily.shape[0])) - 1)),
        "sharpe": float(np.sqrt(252) * daily.mean() / (daily.std() + 1e-12)),
        "max_dd": float((equity / equity.cummax() - 1).min()),
        "turnover_daily": float(weights.diff().abs().sum(axis=1).mean()),
        "max_gross": float(max_gross),
        "lookback": int(lookback),
        "cost_bps": float(cost_bps),
        "seed": int(seed),
    }
    return {"daily": daily, "equity": equity, "weights": weights, "metrics": metrics}
