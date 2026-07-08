"""
Day 25 — Capital base + margin-based returns.

Sharpe is meaningless without a stated denominator (premium vs vega-notional
vs margin give wildly different numbers — SPEC "Capital base").  This module
fixes the denominator: returns are computed on **Reg-T margin**, and the
capital base is the **peak book margin over the trade window** (config
`margin:` block, pre-registered Day 18).

MARGIN MODEL (Reg-T proxy, config: "20% underlying notional +/- OTM adjustment"):
  - Naked short option, per share:
        req = max(0.20*S - OTM, 0.10*S) + premium
    OTM = max(K-S,0) for a call, max(S-K,0) for a put; premium is the current
    mark (buying back the short liability is part of the requirement).
  - Short straddle (both legs short): the standard CBOE rule — the larger of
    the two naked leg requirements, PLUS the premium of the other leg.
  - Long straddle: fully paid; margin = premium debit (call + put).
  Recomputed every bar at S_t and tau_t -> the margin path is PROCYCLICAL
  (rises as the underlying moves against a short and as vol/premium climbs),
  which is exactly the SPEC's "margin expands in stress" interaction.

CAPITAL BASE = max_t (book margin_t).  Returns:
  - numerator: NET book equity path (engine gross equity, Day-24 entry costs
    subtracted at each position's entry bar).  Gross is reported alongside.
  - r_t = d(net equity)_t / capital_base ; cumulative = net_equity_t / base.
Denominator is a single documented scalar so Day-26 Sharpe/tail stats are
comparable.

BOOK: the unit-qty pre-registered book (Day 22/24), NOT the Day-23 risk-sized
portfolio — this keeps the margin/cost/return story on one consistent object.

OUTPUT:
  - data/processed/returns.parquet   (daily: book_margin, net/gross equity, ret)
  - results/returns_summary.json     (tracked, byte-stable)
  - results/plots/margin_returns.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.costs import option_entry_cost, _load_cost_params
from src.backtest.engine import Leg, run_hedged
from src.backtest.reconcile import build_positions, load_price_path
from src.greeks.black_scholes import price_spot

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MULT = 100.0
REG_T_RATE = 0.20        # 20% underlying notional
REG_T_FLOOR = 0.10       # 10% floor
DAYS_PER_YEAR = 365.0


# ── margin model ─────────────────────────────────────────────────────────────

def _naked_leg_margin(S: float, K: float, T: float, sig: float,
                      r: float, q: float, cp: int) -> tuple[float, float]:
    """Reg-T naked short-option requirement per share, and the premium.

    Returns (requirement_per_share, premium_per_share).
    """
    prem = float(price_spot(S, K, max(T, 0.0), sig, r, q, cp))
    otm = max(K - S, 0.0) if cp > 0 else max(S - K, 0.0)
    req = max(REG_T_RATE * S - otm, REG_T_FLOOR * S) + prem
    return req, prem


def straddle_margin(S: float, K: float, T: float, sig: float,
                    r: float, q: float, qty: float) -> float:
    """Reg-T margin for a unit straddle (call+put), in dollars.

    qty < 0 -> short straddle (naked rule); qty > 0 -> long straddle (debit).
    T <= 0 (settled) -> 0.
    """
    if T <= 0.0:
        return 0.0
    req_c, prem_c = _naked_leg_margin(S, K, T, sig, r, q, +1)
    req_p, prem_p = _naked_leg_margin(S, K, T, sig, r, q, -1)
    if qty < 0:
        # larger naked leg + other leg's premium
        margin = (max(req_c, req_p)
                  + (prem_p if req_c >= req_p else prem_c))
    else:
        margin = prem_c + prem_p          # long: fully-paid premium debit
    return margin * MULT * abs(qty)


# ── runner ──────────────────────────────────────────────────────────────────

def run_returns(params: dict | None = None) -> dict:
    """Margin-based return series on the pre-registered unit-qty book."""
    if params is None:
        params = _load_cost_params()

    positions = build_positions()
    path = load_price_path()
    chain = pd.read_parquet(PROCESSED_DIR / "chain_clean.parquet")

    # per-position: engine equity path + margin path + entry cost
    per_pos = []
    for pos in positions:
        win = path[(path["date"] >= pos["date"])
                   & (path["date"] <= pos["expiry"])].reset_index(drop=True)
        legs = [Leg(K=pos["K"], expiry=pos["expiry"], cp=+1, qty=pos["qty"],
                    mark_vol=pos["mark_vol"]),
                Leg(K=pos["K"], expiry=pos["expiry"], cp=-1, qty=pos["qty"],
                    mark_vol=pos["mark_vol"])]
        led = run_hedged(win["date"], win["close"].to_numpy(),
                         legs, r=pos["r"], q=pos["q"])

        # margin path: recompute at each bar's S_t and remaining tau
        margins = []
        for d, s in zip(led["date"], led["S"]):
            tau = max((pos["expiry"] - d).days, 0) / DAYS_PER_YEAR
            margins.append(straddle_margin(s, pos["K"], tau, pos["mark_vol"],
                                           pos["r"], pos["q"], pos["qty"]))
        cost = option_entry_cost(pos, chain, params)["total"]
        per_pos.append({
            "pos": pos,
            "ledger": led.assign(margin=margins),
            "cost": cost,
            "entry_margin": margins[0],
        })

    # ── aggregate onto the union calendar ────────────────────────────────
    all_dates = sorted(set().union(
        *(p["ledger"]["date"].tolist() for p in per_pos)))
    idx = pd.DatetimeIndex(all_dates)

    gross_eq = pd.Series(0.0, index=idx)
    book_margin = pd.Series(0.0, index=idx)
    net_costs = pd.Series(0.0, index=idx)   # cumulative entry cost recognised
    live_ids = [set() for _ in idx]         # which positions are live each bar
    pos_of = {d: i for i, d in enumerate(idx)}
    for j, p in enumerate(per_pos):
        led = p["ledger"].set_index("date")
        eq = led["equity"].reindex(idx, fill_value=0.0)
        final_d = p["ledger"]["date"].iloc[-1]
        eq[idx > final_d] = float(p["ledger"]["equity"].iloc[-1])  # carry settled
        gross_eq = gross_eq + eq

        m = led["margin"].reindex(idx, fill_value=0.0)  # 0 before/after window
        book_margin = book_margin + m

        entry_d = p["pos"]["date"]
        net_costs[idx >= entry_d] += p["cost"]          # cost paid at entry
        for d in p["ledger"]["date"]:                   # membership = window
            live_ids[pos_of[d]].add(j)

    net_eq = gross_eq - net_costs

    capital_base = float(book_margin.max())
    peak_margin_date = str(book_margin.idxmax().date())
    entry_margin = float(book_margin.iloc[0])

    net_ret = net_eq / capital_base
    gross_ret = gross_eq / capital_base
    daily_ret = net_eq.diff().fillna(net_eq.iloc[0]) / capital_base

    # ── procyclicality (genuine, market-driven) ──────────────────────────
    # Two separate effects must NOT be conflated:
    #   (a) the book RAMPS UP as staggered entries add positions — that grows
    #       book margin but is a scheduling artifact, not stress;
    #   (b) on a FIXED live set, spot/vol moves change margin — the real
    #       "SPAN expands in stress" effect the SPEC wants surfaced.
    # Per-position stress ratio isolates (b): each single straddle's peak
    # margin over its own window / its entry margin (pure market move).
    stress_ratios = [float(np.max(p["ledger"]["margin"].to_numpy())
                           / p["entry_margin"])
                     for p in per_pos if p["entry_margin"] > 0]
    stress_ratio_max = float(np.max(stress_ratios))
    stress_ratio_mean = float(np.mean(stress_ratios))
    # corr(dMargin, dEquity) computed ONLY on bars where the live set is
    # unchanged from the prior bar (no entry/settlement jump contaminating dM).
    dm = book_margin.diff().to_numpy()
    de = net_eq.diff().to_numpy()
    clean = np.zeros(len(idx), bool)
    for i in range(1, len(idx)):
        clean[i] = live_ids[i] == live_ids[i - 1]
    mask = clean & np.isfinite(dm) & np.isfinite(de)
    procyc_corr = (float(np.corrcoef(dm[mask], de[mask])[0, 1])
                   if mask.sum() > 2 and np.std(dm[mask]) > 0 else float("nan"))
    # margin at the worst-equity bar vs entry
    trough_date = net_eq.idxmin()
    margin_at_trough = float(book_margin.loc[trough_date])

    out = pd.DataFrame({
        "date": idx,
        "book_margin": book_margin.values,
        "gross_equity": gross_eq.values,
        "net_equity": net_eq.values,
        "net_return": net_ret.values,
        "gross_return": gross_ret.values,
        "daily_return": daily_ret.values,
    })
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED_DIR / "returns.parquet", index=False)

    summary = {
        "capital_base_usd": capital_base,
        "denominator": "peak book Reg-T margin over the trade window "
                       "(config margin.capital_base)",
        "n_positions": len(positions),
        "entry_book_margin_usd": entry_margin,
        "peak_book_margin_usd": capital_base,
        "peak_margin_date": peak_margin_date,
        "peak_driven_by": "max simultaneous position overlap (staggered "
                          "entries), not spot stress - see stress ratios below",
        "gross_pnl_usd": float(gross_eq.iloc[-1]),
        "total_cost_usd": float(net_costs.iloc[-1]),
        "net_pnl_usd": float(net_eq.iloc[-1]),
        "gross_return_on_capital": float(gross_ret.iloc[-1]),
        "net_return_on_capital": float(net_ret.iloc[-1]),
        "per_position_margin_stress_ratio_max": stress_ratio_max,
        "per_position_margin_stress_ratio_mean": stress_ratio_mean,
        "margin_equity_corr_fixed_book": procyc_corr,
        "margin_procyclical": bool(procyc_corr < 0)
        if np.isfinite(procyc_corr) else None,
        "worst_net_equity_usd": float(net_eq.min()),
        "worst_net_equity_date": str(trough_date.date()),
        "book_margin_at_worst_equity_usd": margin_at_trough,
        "date_range": [str(idx[0].date()), str(idx[-1].date())],
        "n_bars": len(out),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "returns_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    _plot(out, summary)

    print(f"returns: capital base ${capital_base:,.0f} (peak margin, "
          f"{peak_margin_date}); net PnL ${summary['net_pnl_usd']:.2f} "
          f"-> {summary['net_return_on_capital']:.2%} on capital")
    print(f"  margin procyclicality (fixed book): per-position stress ratio "
          f"max {stress_ratio_max:.2f}x / mean {stress_ratio_mean:.2f}x, "
          f"corr(dMargin,dEquity) {procyc_corr:+.2f} "
          f"({'procyclical' if procyc_corr < 0 else 'not procyclical'})")
    print(f"-> {RESULTS_DIR / 'returns_summary.json'}")
    return summary


def _plot(df: pd.DataFrame, summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                  height_ratios=[1, 1])
    ax.plot(df["date"], df["net_return"] * 100, color="firebrick",
            label="net return on capital")
    ax.plot(df["date"], df["gross_return"] * 100, color="steelblue", lw=0.9,
            ls="--", label="gross")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("return (% of capital)")
    ax.legend(fontsize=8)
    ax.set_title(f"Day 25 — Margin-based returns "
                 f"(capital base ${summary['capital_base_usd']:,.0f}, "
                 f"net {summary['net_return_on_capital']:.2%})")

    ax2.plot(df["date"], df["book_margin"], color="0.3")
    ax2.axvline(pd.Timestamp(summary["peak_margin_date"]), color="red",
                lw=0.8, ls=":", label="peak margin")
    ax2.set_ylabel("book Reg-T margin ($)")
    ax2.set_xlabel("date")
    ax2.legend(fontsize=8)

    p = PLOTS_DIR / "margin_returns.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    run_returns()
