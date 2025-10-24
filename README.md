Time‑Series Alpha Signals (Enhanced)
===================================

<!-- CI badge -->
[![Real Data Backtests](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/actions/workflows/real_data_backtests.yml/badge.svg?branch=main)](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/actions/workflows/real_data_backtests.yml)

This project implements a small research framework for building and
evaluating cross‑sectional time‑series trading signals.  It expands on
the original momentum prototype with:

- **Purged k‑fold cross‑validation** with optional embargo to avoid
  look‑ahead bias.  A combinatorial variant (CPCV) is also provided to
  assess parameter stability.
- **Event‑based labelling** via the *triple‑barrier* method with
  volatility‑scaled profit‑taking, stop‑loss and vertical timeouts.  Meta‑labels
  indicate when a base classifier is directionally correct.
- **Fractional differentiation** to stationarise non‑stationary price
  series while preserving memory.
- **Statistical metrics** including Newey–West t‑statistics,
  block bootstrap confidence intervals and a simple deflated Sharpe ratio.
- **Evaluation helpers** to produce a distribution of Sharpe ratios
  across cross‑validation folds and summarise performance (CAGR,
  volatility, drawdown, Calmar, hit‑rate, average win/loss, turnover).

The package is fully vectorised, uses pandas for time series
manipulation and includes a command‑line interface for running
single backtests on synthetic or real data.  Advanced model
selection and portfolio construction are left to the user.

> Educational project — not investment advice.

Quickstart
----------

```bash
# Clone & enter
git clone https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal.git
cd Time-Series-Alpha-Signal

# Create a virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run a backtest on synthetic data (momentum signal)
python -m time_series_alpha_signal.cli run --signal momentum --names 10 --days 500 --output tmpdir

# Compute cross‑validated Sharpe distribution (example using default settings)
python -c "import pandas as pd; from time_series_alpha_signal import load_synthetic_prices, cross_validate_sharpe; prices = load_synthetic_prices(10, 500); s = cross_validate_sharpe(prices); print(s)"

# Apply triple‑barrier labels to a single asset
python -c "import pandas as pd; from time_series_alpha_signal import load_synthetic_prices, daily_volatility, triple_barrier_labels; p = load_synthetic_prices(1, 300)['SYM000']; vol = daily_volatility(p); lbl = triple_barrier_labels(p, vol); print(lbl.head())"
```

Directory Structure
-------------------

- **src/time_series_alpha_signal/backtest.py** – core backtesting logic
- **src/time_series_alpha_signal/signals.py** – various cross‑sectional signals
- **src/time_series_alpha_signal/cv.py** – purged k‑fold and CPCV splitters
- **src/time_series_alpha_signal/labels.py** – triple‑barrier labelling and meta‑labels
- **src/time_series_alpha_signal/features.py** – fractional differentiation helpers
- **src/time_series_alpha_signal/metrics.py** – Sharpe, Newey–West, bootstrap and deflated Sharpe
- **src/time_series_alpha_signal/evaluation.py** – cross‑validation evaluation and performance summary
- **src/time_series_alpha_signal/optimizer.py** – leverage constraint helper
- **src/time_series_alpha_signal/cli.py** – basic CLI for running single backtests
- **scripts/run.py** – thin wrapper around the CLI for convenience
