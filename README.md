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


## Results (example)
| Period       | Tickers                        | Cost (bps) | Sharpe (net) | MaxDD  | Avg TO |
|--------------|--------------------------------|------------|--------------|--------|--------|
| 2018–2025    | AAPL, MSFT, GOOGL, AMZN, META | 10         | 0.62         | -13%   | 0.21   |

> Reproduce with:
> ```bash
> tsalpha run --tickers AAPL,MSFT,GOOGL,AMZN,META --start 2018-01-01 --momentum-lb 126 --mr-lb 20 --cost-bps 10 --max-gross 1.0 --out outputs
> ```

### IC & IC Decay
Artifacts under `outputs/`:
- `rank_ic.csv`, `ic_hist.png`
- `ic_decay.csv`, `ic_decay.png`

### Parameter Sweep
```bash
tsalpha sweep --tickers AAPL,MSFT,GOOGL,AMZN,META --start 2018-01-01 \
  --momentum-grid 63 126 189 252 --mr-grid 10 20 40 --cost-bps 10 --max-gross 1.0 \
  --out outputs/sweep
```
Outputs:
- `sweep_results.csv` sorted by Sharpe
- `sweep_sharpe_heatmap.png`


