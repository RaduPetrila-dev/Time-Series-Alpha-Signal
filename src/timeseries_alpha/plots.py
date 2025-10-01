"""
plots.py — simple plotting helpers
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(ec: pd.Series, path: str) -> None:
    plt.figure()
    ec.plot()
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_drawdown(ec: pd.Series, path: str) -> None:
    plt.figure()
    dd = ec / ec.cummax() - 1.0
    dd.plot()
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
