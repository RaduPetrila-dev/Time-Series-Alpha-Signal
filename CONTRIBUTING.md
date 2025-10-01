# Contributing

Thanks for considering a contribution! This project aims to be a clean, minimal reference for vectorized cross-sectional backtesting.

## Ground Rules
- Be respectful and follow my [Code of Conduct](CODE_OF_CONDUCT.md).
- Prefer small, focused PRs.
- Include tests for new behavior and keep coverage healthy.
- Keep functions small, typed, and documented.

## How to Set Up
```bash
git clone https://github.com/RaduPetrila-dev/Time-Series-Alpha-Signals.git
cd Time-Series-Alpha-Signals
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -e .
pip install pytest
pytest -q
