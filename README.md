Time-Series Alpha Signals
=========================
![CI](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/actions/workflows/ci.yml/badge.svg)
[![Real Data Backtests](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/actions/workflows/real_data_backtests.yml/badge.svg?branch=main)](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/actions/workflows/real_data_backtests.yml)

A Python research framework for building, backtesting, and evaluating
cross-sectional time-series trading signals. Implements methodologies
from Lopez de Prado's *Advances in Financial Machine Learning* (2018)
including triple-barrier labelling, purged k-fold cross-validation,
fractional differentiation, and the deflated Sharpe ratio.

> Educational project -- not investment advice.

Features
--------

**Signals** -- 8 cross-sectional signals registered via a pluggable signal registry:
momentum, mean-reversion, EWMA momentum, moving average crossover,
volatility (low-vol tilt), volatility-scaled momentum, regime switch,
and ARIMA forecasts.

**Backtesting** -- production-grade backtest engine with:
- Rebalance frequency control (daily, weekly, monthly).
- Two transaction cost models: proportional (flat bps) and square-root (market impact).
- Drawdown stop mechanism that halts trading after a threshold breach.
- Leverage enforcement via L1-norm scaling.
- Structured `BacktestResult` output with equity curve, weights, and 10+ metrics.

**Cross-validation** -- purged k-fold CV with configurable embargo to prevent
lookahead bias. Combinatorial purged CV (CPCV) for parameter stability analysis.

**Labelling** -- triple-barrier method with volatility-scaled profit-taking
and stop-loss barriers. First-hit-time priority when both barriers are
breached. Meta-labels for bet sizing.

**Meta-labelling model** -- sklearn LogisticRegression with StandardScaler
and L2 regularisation. Three features (momentum, volatility, mean-reversion
z-score). Structured `MetaModelResult` with per-fold accuracy and class balance.

**Statistical metrics** -- annualised Sharpe (ddof=1), Newey-West HAC
t-statistic, block bootstrap confidence intervals with reproducible seeding,
and the probabilistic Sharpe ratio (PSR*) from Bailey & Lopez de Prado (2014)
adjusted for skewness, kurtosis, and multiple testing.

**Fractional differentiation** -- stationarise price series while preserving
memory using the fixed-width window method. Cached weight computation and
automatic minimum-d selection via ADF testing.

**Parameter optimisation** -- generic grid search over any signal parameter
using out-of-sample Sharpe from purged CV. Includes EWMA span convenience wrapper.

**Testing** -- 42 automated tests covering backtest mechanics, CV split
correctness, no-lookahead verification, meta-model training, CLI integration,
and error paths. CI runs lint (ruff), type checking (mypy), and tests across
Python 3.10, 3.11, and 3.12.

Quickstart
----------

```bash
# Clone and install
git clone https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal.git
cd Time-Series-Alpha-Signal
python -m venv .venv && source .venv/bin/activate

# Core install
pip install -e .

# With dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# With ARIMA signal support
pip install -e ".[dev,arima]"
```

Run a backtest on synthetic data:

```bash
python -m time_series_alpha_signal.cli run \
  --signal momentum --names 10 --days 500 \
  --rebalance daily --cost-bps 10 \
  --output results/momentum
```

Run cross-validated Sharpe estimation:

```bash
python -m time_series_alpha_signal.cli cv \
  --signal ewma_momentum --ewma-span 30 \
  --names 10 --days 500 --n-splits 5 \
  --output results/cv
```

Train a meta-labelling model:

```bash
python -m time_series_alpha_signal.cli train \
  --names 5 --days 400 --lookback 10 \
  --vol-span 50 --n-splits 3 \
  --output results/train
```

Fetch real data and run a backtest:

```bash
python scripts/fetch_data.py --universe faang --start 2018-01-01 --end 2023-12-31
python -m time_series_alpha_signal.cli run \
  --csv data/prices_faang_20180101_20231231.csv \
  --signal ewma_momentum --ewma-span 30 \
  --rebalance weekly --impact-model sqrt \
  --output results/faang
```

Generate a parameter sweep heatmap:

```bash
python scripts/heatmap.py \
  --signal momentum \
  --lookbacks 5 10 20 40 60 \
  --grosses 0.5 1.0 1.5 2.0 \
  --output results/sweep
```

Generate benchmark equity curves:

```bash
python scripts/benchmark_eqw.py --csv data/prices_faang_20180101_20231231.csv --type eqw
python scripts/benchmark_eqw.py --csv data/prices_faang_20180101_20231231.csv --type ivol
```

Available Signals
-----------------

| Signal | Function | Key parameter |
|--------|----------|---------------|
| Momentum | `momentum_signal` | `lookback` |
| Mean reversion | `mean_reversion_signal` | `lookback` |
| EWMA momentum | `ewma_momentum_signal` | `span` |
| MA crossover | `moving_average_crossover_signal` | `ma_short`, `ma_long` |
| Volatility | `volatility_signal` | `lookback` |
| Vol-scaled momentum | `volatility_scaled_momentum_signal` | `lookback`, `vol_window` |
| Regime switch | `regime_switch_signal` | `lookback`, `vol_threshold` |
| ARIMA | `arima_signal` | `order` |

Custom signals can be registered at runtime:

```python
from time_series_alpha_signal.backtest import register_signal

def my_signal(prices, lookback=10):
    return prices.pct_change(lookback).shift(1)

register_signal("my_signal", my_signal)
```

Directory Structure
-------------------

```
src/time_series_alpha_signal/
    backtest.py      Signal registry, portfolio construction, simulation engine
    signals.py       8 cross-sectional signal functions
    cv.py            Purged k-fold and combinatorial purged CV splitters
    labels.py        Triple-barrier labelling and meta-labels
    features.py      Fractional differentiation (fracdiff)
    metrics.py       Sharpe, Newey-West, bootstrap CI, deflated Sharpe (PSR*)
    evaluation.py    CV evaluation and fold-level performance summary
    models.py        Meta-labelling model (LogisticRegression + purged CV)
    optimizer.py     Leverage enforcement and generic parameter optimisation
    data.py          Data loaders (synthetic, CSV, Yahoo Finance)
    cli.py           Command-line interface (run, cv, train subcommands)

scripts/
    fetch_data.py       Download prices from Yahoo Finance
    benchmark_eqw.py    Benchmark equity curves (equal-weight, buy-and-hold, inverse-vol)
    heatmap.py          Parameter sweep with Sharpe ratio heatmap
    run.py              Thin CLI wrapper

tests/
    test_cli.py      CLI integration tests (11 tests)
    test_cv.py        Purged CV and cross_validate_sharpe tests (14 tests)
    test_models.py    Meta-model, features, and realised returns tests (14 tests)
```

References
----------

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. H. and Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
  *Journal of Portfolio Management*, 40(5), 94-107.
- Newey, W. K. and West, K. D. (1987). "A Simple, Positive Semi-definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica*, 55(3), 703-708.
