# Time-Series Alpha Signals
![CI](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/actions/workflows/ci.yml/badge.svg)

Vectorised backtesting for cross‑sectional time‑series alpha signals.

This project has grown beyond a simple momentum backtest.  It now supports
several signal types — **momentum**, **mean reversion**, **ARIMA forecasts**, a
**volatility/GARCH proxy**, a **volatility‑scaled momentum** signal, a
**regime‑switching** signal that toggles between momentum and mean reversion
based on market volatility, an **EWMA momentum** signal using an exponential
moving average of returns, and a **moving average crossover** signal based on
the difference between short and long moving averages.  Optional **leverage
caps** and **drawdown stops** allow you to constrain risk, and turnover‑based
**transaction costs** are modelled explicitly.  Data can be loaded from a CSV
file **or fetched automatically from Yahoo Finance** via ``yfinance`` by
specifying tickers and date ranges.  All signals are strictly lagged to
prevent look‑ahead bias and positions are sized via cross‑sectional ranks
with an L1 (gross exposure) normalisation.  A hyper‑parameter sweep script
is included to visualise Sharpe ratios across lookback and gross exposure
grids.

> Educational project — **not** investment advice.

---

## Quickstart

```bash
# 1) Clone & enter
git clone https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal.git
cd Time-Series-Alpha-Signal

# 2) (Recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install package (editable) + test deps
python -m pip install -U pip
pip install -e .
pip install pytest

# 4) Run example pipeline using synthetic data
python scripts/run.py

# 5) Run a custom backtest with different signals (e.g. mean reversion or volatility‑scaled momentum)
python -m time_series_alpha_signal.cli run --signal mean_reversion --names 10 --days 500 --output tmpdir
python -m time_series_alpha_signal.cli run --signal vol_scaled_momentum --names 10 --days 500 --output tmpdir_vs

# Example with the EWMA momentum signal (exponential smoothing)
python -m time_series_alpha_signal.cli run --signal ewma_momentum --ewma-span 30 --names 10 --days 500 --output tmpdir_ewma

# Example with the moving average crossover signal (short vs long trend)
python -m time_series_alpha_signal.cli run --signal ma_crossover --ma-short 10 --ma-long 50 --names 10 --days 500 --output tmpdir_ma

# Example with a 20% drawdown stop and leverage cap using the regime‑switch signal
python -m time_series_alpha_signal.cli run --signal regime_switch --max-drawdown 0.2 --max-leverage 1.5 --names 10 --days 500 --output stopped

# 6) Backtest on your own data from CSV (must contain a Date column and asset prices)
python -m time_series_alpha_signal.cli run --csv path/to/prices.csv --signal momentum --lookback 20 --max-gross 1.0 --cost-bps 10 --output results

# 7) Backtest on real market data using Yahoo Finance (e.g. FAANG stocks 2018–2023)
python -m time_series_alpha_signal.cli run --tickers AAPL,MSFT,GOOGL,META,NFLX --start-date 2018-01-01 --end-date 2023-12-31 --signal vol_scaled_momentum --max-gross 1.0 --cost-bps 10 --output faang_results

# 8) Run tests
pytest -q

# 9) Run a parameter sweep and heatmap (synthetic data)
python scripts/heatmap.py --signal momentum --lookbacks 10 20 40 --grosses 0.5 1.0 1.5 --output sweep

# 10) Run a parameter sweep on your own data (optional)
python scripts/heatmap.py --csv path/to/prices.csv --signal mean_reversion --lookbacks 10 20 40 --grosses 0.5 1.0 1.5 --output my_sweep
```

## Example results

With the extended feature set you can compare different signal types even on
synthetic data.  The table below shows summary metrics from backtests on
randomly generated prices (500 business days, 10 names) using a 20‑day
lookback, unit gross exposure and 10 bps transaction costs.  Results on real
market data will of course differ.

| Signal           | CAGR (annualised) | Sharpe | Max drawdown |
|------------------|------------------:|------:|-------------:|
| momentum         |       −9.55%      | −1.72 | −23.35%      |
| mean reversion   |       −6.13%      | −1.07 | −15.79%      |
| volatility       |       −4.09%      | −0.75 | −13.22%      |
| arima            |      −15.11%      | −2.64 | −28.78%      |
| vol_scaled_momentum |       −9.68%      | −1.76 | −22.72%      |
| regime_switch     |       −9.55%      | −1.72 | −23.35%      |
| ewma_momentum     |      −10.66%      | −1.93 | −23.73%      |
| ma_crossover      |       −3.20%      | −0.56 | −13.08%      |

The synthetic prices follow a random walk, hence the negative returns and
Sharpe ratios.
