
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ann_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    compounded = (1.0 + r).prod()
    years = len(r) / periods_per_year
    if years <= 0:
        return float("nan")
    return float(compounded ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    mu = returns.mean() * periods_per_year
    sd = returns.std(ddof=0) * np.sqrt(periods_per_year)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(mu / sd)


def newey_west_tstat(x: pd.Series, lags: int = 5) -> float:
    x = x.dropna().astype(float)
    T = len(x)
    if T < 3:
        return float("nan")
    mu = x.mean()
    eps = x - mu
    gamma0 = float((eps @ eps) / T)
    s = gamma0
    max_lag = min(lags, T - 1)
    for k in range(1, max_lag + 1):
        cov = float((eps[k:] @ eps[:-k]) / T)
        w = 1.0 - k / (max_lag + 1.0)
        s += 2.0 * w * cov
    se = np.sqrt(s / T)
    if se == 0 or np.isnan(se):
        return float("nan")
    return float(mu / se)


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    fr = prices.pct_change(periods=horizon).shift(-horizon)
    return fr


def _spearman_rank_corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    xr = x[mask].rank(method="average")
    yr = y[mask].rank(method="average")
    xs = xr.std(ddof=0)
    ys = yr.std(ddof=0)
    if xs == 0 or ys == 0:
        return np.nan
    cov = ((xr - xr.mean()) * (yr - yr.mean())).mean()
    return float(cov / (xs * ys))


def rank_ic(signal: pd.DataFrame, fwd_rets: pd.DataFrame) -> pd.Series:
    sig, fr = signal.align(fwd_rets, join="inner")
    idx = sig.index
    ics = []
    for dt in idx:
        ics.append(_spearman_rank_corr(sig.loc[dt], fr.loc[dt]))
    return pd.Series(ics, index=idx, name="rank_ic")


def ic_decay(signal: pd.DataFrame, prices: pd.DataFrame, max_h: int = 10) -> pd.Series:
    vals = []
    for h in range(1, max_h + 1):
        fr = forward_returns(prices, horizon=h)
        ic = rank_ic(signal, fr).mean()
        vals.append(ic)
    return pd.Series(vals, index=range(1, max_h + 1), name="ic_decay")


def plot_ic_histogram(ic: pd.Series, path: str) -> None:
    plt.figure()
    ic.dropna().hist(bins=30)
    plt.title("Rank IC Histogram")
    plt.xlabel("IC")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ic_decay(decay: pd.Series, path: str) -> None:
    plt.figure()
    decay.plot(marker="o")
    plt.title("IC Decay (mean IC vs horizon)")
    plt.xlabel("Horizon (days)")
    plt.ylabel("Mean IC")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
