# Time-Series Alpha Signals
![CI](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/.github/workflows/ci.yml/badge.svg)

Vectorised backtesting for cross‑sectional time‑series alpha signals.

This project has grown beyond a simple momentum backtest.  It now supports
multiple signal types — **momentum**, **mean reversion**, **ARIMA forecasts** and a
**volatility/GARCH proxy** — along with optional leverage **and drawdown
constraints**, transaction cost modelling and the ability to load your own price
data from CSV.  You can also perform hyper‑parameter sweeps and visualise
results via a provided heatmap script.  All signals are strictly lagged to
prevent look‑ahead bias.  Position sizing is done via cross‑sectional ranks
with an L1 (gross exposure) normalisation.

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

# 5) Run a custom backtest with different signals (e.g. mean reversion)
python -m time_series_alpha_signal.cli run --signal mean_reversion --names 10 --days 500 --output tmpdir

# Example with a 20% drawdown stop and leverage cap
python -m time_series_alpha_signal.cli run --signal momentum --max-drawdown 0.2 --max-leverage 1.5 --names 10 --days 500 --output stopped

# 6) Backtest on your own data from CSV (must contain a Date column and asset prices)
python -m time_series_alpha_signal.cli run --csv path/to/prices.csv --signal momentum --lookback 20 --max-gross 1.0 --cost-bps 10 --output results

# 7) Run tests
pytest -q

# 8) Run a parameter sweep and heatmap (synthetic data)
python scripts/heatmap.py --signal momentum --lookbacks 10 20 40 --grosses 0.5 1.0 1.5 --output sweep

# 9) Run a parameter sweep on your own data (optional)
python scripts/heatmap.py --csv path/to/prices.csv --signal mean_reversion --lookbacks 10 20 40 --grosses 0.5 1.0 1.5 --output my_sweep
```

## Example results

The table below shows summary metrics from backtests on
randomly generated prices (500 business days, 10 names) using a 20‑day
lookback, unit gross exposure and 10 bps transaction costs.  Results on real
market data will of course differ.

| Signal           | CAGR (annualised) | Sharpe | Max drawdown |
|------------------|------------------:|------:|-------------:|
| momentum         |       −9.55%      | −1.72 | −23.35%      |
| mean reversion   |       −6.13%      | −1.07 | −15.79%      |
| volatility       |       −4.09%      | −0.75 | −13.22%      |
| arima            |      −15.11%      | −2.64 | −28.78%      |

The synthetic prices follow a random walk, hence the negative returns and
Sharpe ratios.  On real data, you can experiment with lookback windows,
gross exposure limits and different signals to explore alpha opportunities.



