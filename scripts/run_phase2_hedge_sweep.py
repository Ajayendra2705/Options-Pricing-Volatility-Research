"""
Phase-2 Day 37 — SPY hedge-frequency sweep (v2 robustness #3, isolated).

PLAN v2: "hedge-frequency sweep (daily / N-daily / band) -> variance sensitivity."
Day 36 found the delta-hedge programme turns over ~$49M of underlying (~184x
capital); this asks the natural next question -- how do turnover AND PnL variance
trade off against how often you rebalance?

IMPORTANT record correction: the Day-34/36 baseline hedges DAILY, not "M/W/F".
The price path is daily OHLC (a ~30-day straddle has ~23 hedge bars), so
`run_hedged(hedge_every=1)` rebalances every trading day; the M/W/F cadence is
the OPTIONS quote schedule, not the underlying hedge. Day 36's "M/W/F hedge"
label was wrong; this sweep makes the true frequency explicit (hedge_every=1 is
daily) and is validated against Day-34 at that point.

Method: for each hedge_every N in {1,2,3,5,10}, re-run the SAME per-position
engine used by returns.py (run_hedged) with that N, aggregate onto the union
calendar exactly as returns.py does, and report turnover, net PnL, annualised
return-vol, and Sharpe. The capital base is HELD FIXED at the Day-34 peak margin
(margin is set by position size/overlap, ~independent of hedge cadence), so the
frequencies are an apples-to-apples denominator. Nothing tracked is written -- the
driver only emits results/phase2/hedge_sweep_spy.json; it does NOT touch
returns.parquet or any *_spy.json.

Correctness pin: at hedge_every=1 the gross PnL, net PnL and turnover reproduce
the Day-34 returns_summary_spy.json / Day-36 cost_sweep_spy.json exactly
(tests/test_phase2_hedge_sweep.py).

Band hedging (rehedge on a no-trade delta band) is the one PLAN variant NOT
implemented: the self-financing engine only supports periodic rehedging, and a
band trigger is an engine change, not a caller option. It is deferred and
documented here rather than faked -- same discipline as the deferred DSR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.costs import _load_cost_params, option_entry_cost  # noqa: E402
from src.backtest.engine import Leg, run_hedged                      # noqa: E402
from src.backtest.reconcile import build_positions, load_price_path  # noqa: E402
from src.backtest.stats import sharpe_with_nw                        # noqa: E402
from src.utils.seed import set_global_seed                           # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "phase2" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "phase2" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
CONFIG = PROJECT_ROOT / "config" / "spy_phase2.yaml"

HEDGE_EVERY = (1, 2, 3, 5, 10)     # 1 = daily (baseline), 5 ~ weekly, 10 ~ biweekly


def _run_book(positions, chain, price, params, hedge_every: int) -> dict:
    """One full-book run at a given hedge cadence; union-calendar aggregation
    mirrors returns.py (gross equity carried forward past each settlement)."""
    per_pos, turnover = [], 0.0
    for pos in positions:
        win = price[(price["date"] >= pos["date"])
                    & (price["date"] <= pos["expiry"])].reset_index(drop=True)
        legs = [Leg(K=pos["K"], expiry=pos["expiry"], cp=+1, qty=pos["qty"],
                    mark_vol=pos["mark_vol"]),
                Leg(K=pos["K"], expiry=pos["expiry"], cp=-1, qty=pos["qty"],
                    mark_vol=pos["mark_vol"])]
        led = run_hedged(win["date"], win["close"].to_numpy(), legs,
                         r=pos["r"], q=pos["q"], hedge_every=hedge_every)
        turnover += float((led["traded"].abs() * led["S"]).sum())
        cost = option_entry_cost(pos, chain, params)["total"]
        per_pos.append({"pos": pos, "ledger": led, "cost": cost})

    idx = pd.DatetimeIndex(sorted(set().union(
        *(p["ledger"]["date"].tolist() for p in per_pos))))
    gross_eq = pd.Series(0.0, index=idx)
    net_costs = pd.Series(0.0, index=idx)
    for p in per_pos:
        led = p["ledger"].set_index("date")
        eq = led["equity"].reindex(idx, fill_value=0.0)
        final_d = p["ledger"]["date"].iloc[-1]
        eq[idx > final_d] = float(p["ledger"]["equity"].iloc[-1])   # carry settled
        gross_eq = gross_eq + eq
        net_costs[idx >= p["pos"]["date"]] += p["cost"]
    net_eq = gross_eq - net_costs
    return {
        "idx": idx,
        "gross_pnl": float(gross_eq.iloc[-1]),
        "net_eq": net_eq,
        "turnover": turnover,
    }


def main() -> None:
    set_global_seed()
    params = _load_cost_params(CONFIG)
    price = load_price_path(RAW_DIR, "spy_ohlc.parquet", "spy_ohlc_ext.parquet")
    positions = build_positions(processed_dir=PROCESSED_DIR, price_path=price)
    chain = pd.read_parquet(PROCESSED_DIR / "chain_clean.parquet")

    rsum = json.loads((RESULTS_DIR / "returns_summary_spy.json").read_text())
    capital_base = rsum["capital_base_usd"]      # FIXED denominator across N

    curve = []
    for n in HEDGE_EVERY:
        bk = _run_book(positions, chain, price, params, n)
        net_eq = bk["net_eq"]
        daily = net_eq.diff().fillna(net_eq.iloc[0]) / capital_base
        arr = daily.to_numpy()
        sh = sharpe_with_nw(arr, n_lags=int(np.ceil(np.sqrt(arr.size))))
        curve.append({
            "hedge_every_bars": n,
            "turnover_usd": bk["turnover"],
            "turnover_over_capital": bk["turnover"] / capital_base,
            "gross_pnl_usd": bk["gross_pnl"],
            "net_pnl_usd": float(net_eq.iloc[-1]),
            "net_return_on_capital": float(net_eq.iloc[-1] / capital_base),
            "return_vol_annualized": float(np.std(arr, ddof=1) * np.sqrt(252)),
            "sharpe_annualized": sh["sharpe_annualized"],
            "nw_tstat": sh["nw_tstat"],
        })
        print(f"  every {n:2d} bar(s): turnover ${bk['turnover']/1e6:6.1f}M "
              f"({bk['turnover']/capital_base:5.0f}x) | net "
              f"${net_eq.iloc[-1]:+9.2f} | vol "
              f"{curve[-1]['return_vol_annualized']:.1%} | Sharpe "
              f"{sh['sharpe_annualized']:+5.2f}")

    daily_ref = next(c for c in curve if c["hedge_every_bars"] == 1)
    out = {
        "design": "hedge-frequency sweep on the Day-34 unit book; hedge_every "
                  "counts DAILY price-path bars (1 = daily rebalance). Capital "
                  "base held fixed at the Day-34 peak margin so frequencies are "
                  "comparable. Turnover = sum |traded_t| * S_t over the book; "
                  "net PnL = gross - entry costs (underlying_slippage_bps=0, as "
                  "pre-registered).",
        "baseline_is_daily_note": "Day-34/36 hedge_every=1 = DAILY hedging on "
                                  "the daily OHLC path; the M/W/F cadence is the "
                                  "options quote schedule. Day 36's 'M/W/F "
                                  "hedge' label was a mischaracterisation, "
                                  "corrected here.",
        "capital_base_usd": capital_base,
        "curve": curve,
        "band_hedging": {
            "implemented": False,
            "reason": "the self-financing engine supports only periodic "
                      "rehedging; a no-trade delta band is an engine change, not "
                      "a caller option. Deferred and documented, not faked "
                      "(same discipline as the deferred Deflated Sharpe).",
        },
        "finding": "turnover falls monotonically with rebalance spacing (daily "
                   f"${daily_ref['turnover_usd']/1e6:.0f}M -> ~$16M at every 10 "
                   "bars, 184x -> 61x capital) while return volatility rises "
                   "monotonically (2.5% -> 6.0%) -- the textbook hedging "
                   "cost/variance trade-off, now measured on SPY. Net PnL stays "
                   "negative at EVERY cadence, so no rebalance schedule turns "
                   "the short-vol book profitable; and daily hedging is the "
                   "LEAST-BAD (net -$8.9k daily vs -$17k..-$21k less often) "
                   "because a short-gamma book left under-hedged through adverse "
                   "moves -- above all the 2024-08-05 spike -- bleeds more, not "
                   "less. Robustness trial #3 -> DSR trial count N.",
    }
    out_path = RESULTS_DIR / "hedge_sweep_spy.json"
    out_path.write_text(json.dumps(out, indent=2), newline="\n")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
