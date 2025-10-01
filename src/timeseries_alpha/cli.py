from __future__ import annotations

import argparse
import os
from datetime import date
from typing import List

import numpy as np
import pandas as pd

from timeseries_alpha.data import compute_returns, load_prices
from timeseries_alpha.signals import momentum_signal, mean_reversion_zscore, combine_signals
from timeseries_alpha.backtest import backtest
from timeseries_alpha.metrics import sharpe, max_drawdown, equity_curve, avg_turnover
from timeseries_alpha.analytics import forward_returns, rank_ic, ic_decay, plot_ic_histogram, plot_ic_decay


def cmd_run(args: argparse.Namespace) -> None:
    tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start = args.start
    end = args.end or str(date.today())

    os.makedirs(args.out, exist_ok=True)

    prices = load_prices(tickers, start, end)
    rets = compute_returns(prices, method="simple")

    # Signals
    sigs = []
    if args.momentum_lb:
        sigs.append(momentum_signal(prices, args.momentum_lb))
    if args.mr_lb:
        sigs.append(mean_reversion_zscore(rets, args.mr_lb))
    if not sigs:
        raise SystemExit("No signals specified. Use --momentum-lb and/or --mr-lb.")

    sig = combine_signals(sigs)

    bt = backtest(prices, sig, cost_bps=args.cost_bps, max_gross=args.max_gross, lag_signal=1)

    ec = equity_curve(bt["net_returns"], start_value=1.0)
    s = sharpe(bt["net_returns"])
    mdd = max_drawdown(ec)
    to = avg_turnover(bt["turnover"])

    # Analytics: IC and IC decay
    fr = forward_returns(prices, horizon=1)
    ic = rank_ic(sig, fr)
    decay = ic_decay(sig, prices, max_h=10)

    # Save outputs
    ec.to_frame("equity").to_csv(os.path.join(args.out, "equity.csv"))
    bt["net_returns"].to_csv(os.path.join(args.out, "net_returns.csv"), header=["net"])
    ic.to_csv(os.path.join(args.out, "rank_ic.csv"), header=["ic"])
    decay.to_csv(os.path.join(args.out, "ic_decay.csv"), header=["mean_ic"])

    from .plots import plot_equity_curve, plot_drawdown
    plot_equity_curve(ec, os.path.join(args.out, "equity_curve.png"))
    plot_drawdown(ec, os.path.join(args.out, "drawdown.png"))
    plot_ic_histogram(ic, os.path.join(args.out, "ic_hist.png"))
    plot_ic_decay(decay, os.path.join(args.out, "ic_decay.png"))

    # Summary
    summary = f"""# Run Summary
**Tickers:** {", ".join(tickers)}
**Period:** {start} → {end}

## Performance
- Sharpe (net): {s:.2f}
- Max Drawdown: {mdd:.2%}
- Avg Daily Turnover: {to:.2f}
- Costs (bps): {args.cost_bps}

## Signals
- Momentum LB: {args.momentum_lb if args.momentum_lb else "-"}
- Mean-Reversion LB: {args.mr_lb if args.mr_lb else "-"}

## Rank IC (1-day)
- Mean IC: {ic.mean():.3f}
- Median IC: {ic.median():.3f}
"""
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(summary)

    print("Done. Artifacts in:", args.out)


def cmd_sweep(args: argparse.Namespace) -> None:
    tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    start = args.start
    end = args.end or str(date.today())
    os.makedirs(args.out, exist_ok=True)

    prices = load_prices(tickers, start, end)
    rets = compute_returns(prices, method="simple")

    rows = []
    for lb_m in args.momentum_grid:
        mom = momentum_signal(prices, lb_m) if lb_m > 0 else None
        for lb_r in args.mr_grid:
            mr = mean_reversion_zscore(rets, lb_r) if lb_r > 0 else None
            sigs = [x for x in [mom, mr] if x is not None]
            if not sigs:
                continue
            sig = combine_signals(sigs)
            bt = backtest(prices, sig, cost_bps=args.cost_bps, max_gross=args.max_gross, lag_signal=1)
            s = sharpe(bt["net_returns"])
            ec = equity_curve(bt["net_returns"])
            from .metrics import max_drawdown as mdd_fn
            mdd = mdd_fn(ec)
            to = avg_turnover(bt["turnover"])
            # 1-day IC
            ic = rank_ic(sig, forward_returns(prices, horizon=1)).mean()

            rows.append({
                "momentum_lb": lb_m,
                "mr_lb": lb_r,
                "sharpe": s,
                "max_drawdown": mdd,
                "avg_turnover": to,
                "mean_ic_1d": ic,
            })

    df = pd.DataFrame(rows).sort_values(["sharpe"], ascending=False)
    csv_path = os.path.join(args.out, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print("Wrote:", csv_path)

    # Optional heatmap (Sharpe)
    try:
        import matplotlib.pyplot as plt
        pivot = df.pivot(index="momentum_lb", columns="mr_lb", values="sharpe")
        plt.figure()
        im = plt.imshow(pivot, origin="lower", aspect="auto")
        plt.title("Sharpe Heatmap (rows=Momentum LB, cols=MR LB)")
        plt.xlabel("MR Lookback")
        plt.ylabel("Momentum Lookback")
        plt.colorbar(im, label="Sharpe")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.tight_layout()
        heat_path = os.path.join(args.out, "sweep_sharpe_heatmap.png")
        plt.savefig(heat_path)
        plt.close()
        print("Wrote:", heat_path)
    except Exception as e:
        print("Heatmap generation skipped:", e)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tsalpha", description="Time-Series Alpha CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run", help="Run example pipeline")
    p_run.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL,AMZN,META")
    p_run.add_argument("--start", type=str, default="2018-01-01")
    p_run.add_argument("--end", type=str, default=None)
    p_run.add_argument("--momentum-lb", type=int, default=126)
    p_run.add_argument("--mr-lb", type=int, default=20)
    p_run.add_argument("--cost-bps", type=float, default=10.0)
    p_run.add_argument("--max-gross", type=float, default=1.0)
    p_run.add_argument("--out", type=str, default="outputs")
    p_run.set_defaults(func=cmd_run)

    # sweep
    p_sw = sub.add_parser("sweep", help="Parameter sweep over lookbacks")
    p_sw.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOGL,AMZN,META")
    p_sw.add_argument("--start", type=str, default="2018-01-01")
    p_sw.add_argument("--end", type=str, default=None)
    p_sw.add_argument("--momentum-grid", type=int, nargs="+", default=[63, 126, 189, 252])
    p_sw.add_argument("--mr-grid", type=int, nargs="+", default=[10, 20, 40])
    p_sw.add_argument("--cost-bps", type=float, default=10.0)
    p_sw.add_argument("--max-gross", type=float, default=1.0)
    p_sw.add_argument("--out", type=str, default="outputs/sweep")
    p_sw.set_defaults(func=cmd_sweep)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
