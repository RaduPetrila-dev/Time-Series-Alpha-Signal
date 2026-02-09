# Time-Series Alpha Signal

A Python framework for building, backtesting, and evaluating time-series alpha signals with walk-forward validation and risk-adjusted portfolio construction.

Built on the methodology from Lopez de Prado (2018) *Advances in Financial Machine Learning* and Bailey & Lopez de Prado (2014) *The Deflated Sharpe Ratio*.

## Features

- **8 alpha signals** with a pluggable signal registry: momentum, mean reversion, EWMA momentum, volatility, vol-scaled momentum, regime switch, moving average crossover, ARIMA (optional).
- **Signal combination engine** with equal-weight blending and walk-forward optimised blending via grid search over weight combinations.
- **Risk model** with inverse-volatility weighting, volatility targeting, signal smoothing, and signal clipping for risk-adjusted portfolio construction.
- **Walk-forward validation** with rolling train/test windows. No lookahead bias. Out-of-sample returns stitched together for realistic performance estimates.
- **Triple barrier labelling** and meta-labelling with sklearn-based logistic regression, StandardScaler, and L2 regularisation.
- **Purged K-Fold cross-validation** and combinatorial purged CV to prevent information leakage between train/test splits.
- **Deflated Sharpe Ratio (PSR\*)** from Bailey & Lopez de Prado (2014), Newey-West t-statistics, and block bootstrap confidence intervals.
- **Rebalance frequency control**: daily, weekly, or monthly.
- **Two transaction cost models**: proportional and square-root market impact.
- **Structured outputs**: `BacktestResult` and `MetaModelResult` dataclasses with `.to_dict()` serialisation.
- **76+ automated tests** across all modules.
- **CI/CD**: lint (ruff), typecheck (mypy), test (pytest) on Python 3.10-3.12, plus automated real-data backtests via GitHub Actions.

## Quickstart

```bash
# Core install
pip install -e .

# With dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# With ARIMA signal support
pip install -e ".[arima]"

# Everything
pip install -e ".[dev,arima]"
```

## CLI Usage

Run a single-signal backtest:

```bash
tsalpha -v run \
  --tickers AAPL,AMZN,GOOGL,META,NFLX \
  --start-date 2018-01-01 --end-date 2023-12-31 \
  --signal ewma_momentum --ewma-span 30 \
  --max-gross 1.0 --cost-bps 10 \
  --rebalance daily --impact-model proportional \
  --output results/faang_ewma
```

Run a combined signal backtest with risk model:

```bash
python scripts/run_combined.py \
  --tickers AAPL,AMZN,GOOGL,META,NFLX \
  --start 2018-01-01 --end 2023-12-31 \
  --signals momentum,mean_reversion,ewma_momentum \
  --mode equal_weight \
  --risk-model --vol-target 0.10 --smooth-halflife 5 \
  --cost-bps 10 --max-gross 1.0 \
  --output results/faang_risk -v
```

Run walk-forward optimisation on crypto:

```bash
python scripts/run_combined.py \
  --tickers BTC-USD,ETH-USD \
  --start 2018-01-01 --end 2023-12-31 \
  --signals momentum,mean_reversion,ewma_momentum,volatility \
  --mode walk_forward \
  --train-days 504 --test-days 252 --step-days 252 \
  --risk-model --vol-target 0.15 --clip-zscore 2.0 \
  --cost-bps 20 --max-gross 1.0 \
  --output results/crypto_wf -v
```

## Available Signals

| Signal | Function | Key Parameters |
|---|---|---|
| Momentum | `momentum_signal` | `lookback` |
| Mean Reversion | `mean_reversion_signal` | `lookback` |
| EWMA Momentum | `ewma_momentum_signal` | `span` |
| Volatility | `volatility_signal` | `lookback` |
| Vol-Scaled Momentum | `volatility_scaled_momentum_signal` | `lookback`, `vol_window` |
| Regime Switch | `regime_switch_signal` | `lookback`, `vol_window`, `vol_threshold` |
| MA Crossover | `moving_average_crossover_signal` | `ma_short`, `ma_long` |
| ARIMA | `arima_signal` | `arima_order` (requires `.[arima]`) |

### Custom Signal Registration

```python
from time_series_alpha_signal import signals

@signals.register_signal("my_signal")
def my_signal(prices, lookback=20):
    return prices.pct_change(lookback).rank(axis=1, pct=True) - 0.5
```

## Risk Model

The risk model transforms raw signal scores into risk-adjusted portfolio weights through a composable pipeline:

1. **Signal clipping** caps extreme z-scores (default 2.0 std) to reduce concentration risk.
2. **Signal smoothing** applies exponential decay (default 5-day halflife) to reduce turnover and save transaction costs.
3. **Inverse-volatility weighting** scales each position by 1/realised vol for approximate equal risk contribution across assets.
4. **Volatility targeting** scales the entire portfolio to a target annualised volatility (e.g. 10%), reducing exposure when vol spikes.

```python
from time_series_alpha_signal.risk_model import RiskConfig, apply_risk_model

config = RiskConfig(
    clip_zscore=2.0,
    smooth_halflife=5,
    inverse_vol=True,
    vol_target=0.10,
    max_leverage=2.0,
)

weights = apply_risk_model(signal, prices, config=config)
```

All transforms are optional and individually configurable via `RiskConfig`.

## Signal Combination

### Equal-Weight Blend

Z-score normalises each signal cross-sectionally and averages them. No fitting required.

```python
from time_series_alpha_signal.combiner import blend_equal_weight
from time_series_alpha_signal.risk_model import RiskConfig

result = blend_equal_weight(
    prices,
    signal_names=["momentum", "mean_reversion", "ewma_momentum"],
    risk_config=RiskConfig(vol_target=0.10),
)
print(result.metrics)
```

### Walk-Forward Optimised Blend

Splits history into rolling 2-year train / 1-year test windows. On each train window, grid-searches all weight combinations and selects the highest Sharpe. Applies the winning weights out-of-sample.

```python
from time_series_alpha_signal.combiner import walk_forward_optimize

wf = walk_forward_optimize(
    prices,
    signal_names=["momentum", "mean_reversion", "ewma_momentum", "volatility"],
    train_days=504,
    test_days=252,
    risk_config=RiskConfig(vol_target=0.10),
)
print(wf.overall_metrics)
for w in wf.window_results:
    print(f"W{w['window']}: train SR={w['train_sharpe']:.2f}, OOS SR={w['oos_sharpe']:.2f}")
```

## Scripts

| Script | Description |
|---|---|
| `scripts/run_combined.py` | CLI for combined signal backtests with risk model |
| `scripts/fetch_data.py` | Download historical prices from Yahoo Finance |
| `scripts/heatmap.py` | Parameter sweep with Sharpe heatmap visualisation |
| `scripts/benchmark_eqw.py` | Benchmark equity curves (equal-weight, buy-and-hold, inverse-vol) |

## Directory Structure

```
src/time_series_alpha_signal/
    __init__.py         Public API exports
    backtest.py         Backtest engine with rebalance and cost models
    cli.py              Command-line interface
    combiner.py         Signal combination (equal-weight, walk-forward)
    cv.py               Purged K-Fold and combinatorial purged CV
    data.py             Data loaders (synthetic, CSV, Yahoo Finance)
    evaluation.py       Cross-validation evaluation and fold metrics
    features.py         Fractional differentiation (fracdiff)
    labels.py           Triple barrier and meta-labelling
    metrics.py          Sharpe, PSR*, Newey-West, bootstrap CI
    models.py           Meta-model training and evaluation
    optimizer.py        Parameter optimisation and leverage constraints
    risk_model.py       Risk model (inverse-vol, vol targeting, smoothing)
    signals.py          Signal registry and 8 signal implementations
scripts/
    run_combined.py     Combined signal backtest CLI
    fetch_data.py       Yahoo Finance data downloader
    heatmap.py          Parameter sweep heatmap
    benchmark_eqw.py    Benchmark comparisons
tests/
    test_backtest.py    Backtest engine tests
    test_combiner.py    Signal combination tests
    test_cv.py          Cross-validation tests
    test_labels.py      Triple barrier tests
    test_models.py      Meta-model tests
    test_risk_model.py  Risk model tests
    test_signals.py     Signal tests
```

## Development

```bash
# Run tests
pytest -v

# Run linter
ruff check src/ tests/ scripts/

# Run type checker
mypy src/

# Format code
ruff format src/ tests/ scripts/
```

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*, 40(5).
- Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3).
