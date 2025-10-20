# Time-Series Alpha Signals
![CI](https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signal/actions/workflows/ci.yml/badge.svg)

Vectorized backtesting for cross-sectional time-series alpha signals with:
- **No-lookahead** (signals lagged before weights)
- **L1 exposure control** (Σ|weights| = max_gross)
- **Turnover-based transaction costs**
- Clean, **pandas**/**numpy** vectorization and robust index/NaN handling

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

# 4) Run example pipeline
python scripts/run.py

# 5) Run tests
pytest -q
```

## Example results

Running the package against synthetic data demonstrates how the momentum strategy behaves under different universe sizes.  The table below shows key metrics from backtests on randomly generated prices (500 business days) for 5, 10 and 20 names, using a 20‑day lookback, unit gross exposure and 10 bps transaction costs.

| Universe size | CAGR        | Sharpe | Max drawdown |
|-------------:|-----------:|-------:|-------------:|
| 5 names      | -16.67%     | -2.24  | -32.00%      |
| 10 names     | -8.04%      | -1.47  | -16.13%      |
| 20 names     | -8.46%      | -2.27  | -16.54%      |

These values come from the synthetic example; performance on real market data will differ.  The negative metrics reflect the fact that the random price series have no intrinsic momentum signal.


