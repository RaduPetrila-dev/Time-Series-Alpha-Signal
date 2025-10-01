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

