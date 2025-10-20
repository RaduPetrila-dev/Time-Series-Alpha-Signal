import pandas as pd
from pathlib import Path

prices = pd.read_csv("data/prices_faang_2018_2023.csv", parse_dates=["Date"], index_col="Date")
rets = prices.pct_change().dropna()
eqw_rets = rets.mean(axis=1)
eqw_equity = (1 + eqw_rets).cumprod()
Path("results/benchmark").mkdir(parents=True, exist_ok=True)
eqw_equity.to_csv("results/benchmark/faang_eqw_equity.csv")
print("Saved results/benchmark/faang_eqw_equity.csv", eqw_equity.shape)
