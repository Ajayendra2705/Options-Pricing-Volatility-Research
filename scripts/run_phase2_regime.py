"""
Phase-2 Day 35 — SPY vol-regime split (v2 robustness appendix, isolated).

The Day-34 walk-forward is a null (net -$8,936, Sharpe insignificant) whose whole
shape is one crash: three positive test folds, then a -$11,656 settlement tail
carrying the 2024-08-05 VIX-65 spike. The pre-registered robustness question
(PLAN v2, "regime split — does edge survive high-vol?") is therefore the sharpest
one to ask of THIS study: split the book by volatility regime and see where the
PnL actually lives.

Regime variable: trailing 21-day Yang-Zhang realized vol (the study's own
estimator, src.backtest.realized_vol.yang_zhang_vol), computed on the FULL price
path (base + ext, through 2024-08-30) so the crash tail is covered. It is
backward-looking by construction (the estimate at t uses bars t-20..t), so
bucketing a day/position by it uses only information available by that day's
close. It is a self-contained, no-new-data, no-lookahead vol axis; for SPY it
tracks VIX ~1:1 (VIX is 30-day implied on SPX), and the cross-check against the
study's entry ATM IV is reported.

    src.backtest.realized_vol.yang_zhang_vol   (regime variable)
    data/phase2/processed/returns.parquet      (day-level net return series)
    results/phase2/attribution_reconcile_spy.json  (position-level gross PnL)
      -> results/phase2/regime_split_spy.json

Two cuts, each honest about a different question:
  - DAY-LEVEL (283 net-return days): per-regime net PnL / Sharpe / NW t. This is
    the rigorous cut; buckets partition the days and reconcile to the -$8,936
    net total exactly. Answers "on which days does the book make or lose money?"
  - ENTRY-LEVEL (294 positions): gross PnL grouped by the vol regime at ENTRY.
    Answers "does selling into calm vol pay?" -- and it is where the crash story
    inverts: the tail's losers were SOLD in calm June, so they land in the LOW
    bucket, not the high one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.realized_vol import yang_zhang_vol          # noqa: E402
from src.backtest.stats import sharpe_with_nw                 # noqa: E402
from src.utils.seed import set_global_seed                    # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "phase2" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "phase2" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"

RV_WINDOW = 21
LABELS = ("low_vol", "mid_vol", "high_vol")


def _terciles(values: pd.Series) -> tuple[pd.Series, list[float]]:
    """Assign low/mid/high by the 1/3 and 2/3 quantiles of `values`."""
    lo, hi = float(values.quantile(1 / 3)), float(values.quantile(2 / 3))
    label = pd.cut(values, bins=[-np.inf, lo, hi, np.inf], labels=list(LABELS))
    return label, [lo, hi]


def _day_block(r: pd.Series, capital_base: float, regime: pd.Series) -> dict:
    arr = r.to_numpy()
    sh = sharpe_with_nw(arr, n_lags=int(np.ceil(np.sqrt(arr.size))))
    return {
        "n_days": int(arr.size),
        "net_pnl_usd": float(arr.sum() * capital_base),
        "sharpe_annualized": sh["sharpe_annualized"],
        "nw_tstat": sh["nw_tstat"],
        "worst_day_usd": float(arr.min() * capital_base),
        "best_day_usd": float(arr.max() * capital_base),
        "mean_trailing_rv": float(regime.loc[r.index].mean()),
    }


def main() -> None:
    set_global_seed()

    # ── regime variable on the full OHLC path (covers the settlement tail) ────
    # Yang-Zhang needs OHLC; load_price_path returns close-only (the hedge
    # engine's need), so read the two raw OHLC files directly and concat.
    ohlc = pd.concat([pd.read_parquet(RAW_DIR / "spy_ohlc.parquet"),
                      pd.read_parquet(RAW_DIR / "spy_ohlc_ext.parquet")],
                     ignore_index=True)
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    ohlc = ohlc.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    last_bar = ohlc["date"].max()
    yz = yang_zhang_vol(ohlc[["date", "open", "high", "low", "close"]], RV_WINDOW)
    yz.index = pd.to_datetime(yz.index)

    rsum = json.loads((RESULTS_DIR / "returns_summary_spy.json").read_text())
    capital_base = rsum["capital_base_usd"]

    # ── DAY-LEVEL cut: net daily returns by that day's vol tercile ────────────
    ret = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.set_index("date")["daily_return"]
    reg_day = yz.reindex(daily.index)
    if reg_day.isna().any():
        raise ValueError(f"{int(reg_day.isna().sum())} P&L days lack a trailing "
                         "RV value — price path does not cover them")
    day_label, day_edges = _terciles(reg_day)

    day_level = {lab: _day_block(daily[day_label == lab], capital_base, reg_day)
                 for lab in LABELS}
    all_days = _day_block(daily, capital_base, reg_day)

    # ── ENTRY-LEVEL cut: gross position PnL by the vol regime at ENTRY ────────
    rec = json.loads((RESULTS_DIR / "attribution_reconcile_spy.json").read_text())
    pos = pd.DataFrame(rec["positions"])
    pos["date"] = pd.to_datetime(pos["date"])
    entry_rv = yz.reindex(pd.DatetimeIndex(pos["date"].unique()))
    if entry_rv.isna().any():
        raise ValueError("an entry date lacks a trailing RV value")
    entry_label, entry_edges = _terciles(entry_rv)
    pos["regime"] = pos["date"].map(entry_label)

    entry_level = {}
    for lab in LABELS:
        b = pos[pos["regime"] == lab]
        entry_level[lab] = {
            "n_positions": int(len(b)),
            "gross_pnl_usd": float(b["pnl"].sum()),
            "mean_entry_trailing_rv": float(entry_rv[entry_label == lab].mean()),
        }

    # cross-check: does the RV regime track the implied-vol (VIX-like) regime?
    sig = pd.read_parquet(PROCESSED_DIR / "signal.parquet")
    sig["date"] = pd.to_datetime(sig["date"])
    atm_by_date = sig.groupby("date")["atm_iv"].mean()
    join = pd.concat([entry_rv.rename("rv"),
                      atm_by_date.reindex(entry_rv.index).rename("atm_iv")],
                     axis=1).dropna()
    rv_atm_corr = float(join["rv"].corr(join["atm_iv"]))

    out = {
        "regime_variable": {
            "name": f"trailing {RV_WINDOW}-day Yang-Zhang realized vol",
            "source": "src.backtest.realized_vol.yang_zhang_vol on the full "
                      "OHLC path (spy_ohlc + spy_ohlc_ext, through "
                      f"{last_bar.date()})",
            "backward_looking": True,
            "note": "estimate at t uses bars t-20..t, so bucketing day/position "
                    "t uses only information available by t's close; no lookahead",
            "vix_proxy_cross_check": {
                "corr_trailing_rv_vs_entry_atm_iv": rv_atm_corr,
                "note": "SPY entry ATM IV is the study's implied-vol (VIX-like) "
                        "level; a high correlation means the realized-vol regime "
                        "and the implied-vol regime pick out the same days",
            },
        },
        "capital_base_usd": capital_base,
        "day_level": {
            "question": "on which days does the book make or lose money?",
            "reconciles_to": "net PnL over all 283 return days (incl. burn-in "
                             "and settlement tail)",
            "tercile_edges_rv": {"low|mid": day_edges[0], "mid|high": day_edges[1]},
            "regimes": day_level,
            "all_days": all_days,
        },
        "entry_level": {
            "question": "does selling into a given vol regime pay? (gross, by "
                        "the regime at ENTRY)",
            "reconciles_to": "gross PnL over all 294 positions",
            "finding": "the loss concentrates in the MID-vol entry bucket "
                       "(~11% entry RV): those positions ran into the "
                       "2024-08-05 spike in their settlement tail. Entries made "
                       "when vol was ALREADY high were not the losers, so the "
                       "damage was not avoidable by refusing to sell in high "
                       "vol — it came from selling ordinary vol that then "
                       "spiked. This is the entry-timing complement to the "
                       "day-level cut, where the loss is monotone in the "
                       "contemporaneous regime.",
            "tercile_edges_rv": {"low|mid": entry_edges[0],
                                 "mid|high": entry_edges[1]},
            "regimes": entry_level,
        },
    }
    out_path = RESULTS_DIR / "regime_split_spy.json"
    out_path.write_text(json.dumps(out, indent=2), newline="\n")

    # ── console ───────────────────────────────────────────────────────────────
    print(f"regime variable: trailing {RV_WINDOW}d Yang-Zhang RV "
          f"(corr vs entry ATM IV = {rv_atm_corr:+.2f})")
    print(f"\nDAY-LEVEL (net, by that day's vol tercile; edges "
          f"{day_edges[0]:.1%} / {day_edges[1]:.1%}):")
    for lab in LABELS:
        b = day_level[lab]
        print(f"  {lab:9s}: net ${b['net_pnl_usd']:+9.2f} | Sharpe "
              f"{b['sharpe_annualized']:+5.2f} | NW t {b['nw_tstat']:+5.2f} "
              f"| {b['n_days']} days | mean RV {b['mean_trailing_rv']:.1%}")
    a = all_days
    print(f"  {'ALL':9s}: net ${a['net_pnl_usd']:+9.2f} | Sharpe "
          f"{a['sharpe_annualized']:+5.2f} | NW t {a['nw_tstat']:+5.2f}")
    csum = sum(day_level[l]["net_pnl_usd"] for l in LABELS)
    print(f"  reconcile: sum(regimes) {csum:+.2f} vs ALL "
          f"{a['net_pnl_usd']:+.2f} | diff {csum - a['net_pnl_usd']:+.6f}")

    print(f"\nENTRY-LEVEL (gross, by vol regime at entry; edges "
          f"{entry_edges[0]:.1%} / {entry_edges[1]:.1%}):")
    for lab in LABELS:
        b = entry_level[lab]
        print(f"  {lab:9s}: gross ${b['gross_pnl_usd']:+9.2f} | "
              f"{b['n_positions']:3d} positions | mean entry RV "
              f"{b['mean_entry_trailing_rv']:.1%}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
