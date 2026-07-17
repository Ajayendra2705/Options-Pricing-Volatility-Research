"""
Phase-2 Day 34 — SPY walk-forward backtest, isolated from v1.

Same code as v1's Days 22-28 backtest chain, pointed at Phase-2 paths via the
Day 32/34 seams, in v1's order and with v1's rule: the attribution
reconciliation GATE runs first, and nothing downstream executes if it fails —
no PnL claim before the Greeks decomposition closes.

    data/phase2/processed/{signal,forwards,svi_params_joint,chain_clean}.parquet
    data/phase2/raw/spy_ohlc(.parquet|_ext.parquet)      (hedge/settlement path)
      -> reconcile GATE -> portfolio -> costs -> returns -> metrics -> stats
      -> results/phase2/{attribution_reconcile,portfolio_summary,costs_summary,
                         returns_summary,metrics,walkforward}_spy.json

Config: config/spy_phase2.yaml (sizing + costs blocks; values are literal
copies of primary.yaml per the pre-registration, but the run must read its own
pre-registration, not v1's).

Walk-forward (design block of spy_phase2.yaml): nothing in the primary chain is
fit on the window (HAR is expanding-OOS by construction; sizing/cost rules are
fixed), so the folds are honest evaluation slices, not refit boundaries. The
fold report carves the daily net return series into the pre-registered test
folds (2023Q4, 2024Q1, 2024Q2; Q3-2023 burn-in) and reports per-fold PnL +
Sharpe + NW t, so the aggregate cannot hide one lucky quarter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.costs import run_costs                     # noqa: E402
from src.backtest.metrics import run_metrics                 # noqa: E402
from src.backtest.portfolio import run_portfolio             # noqa: E402
from src.backtest.reconcile import load_price_path, run_reconcile  # noqa: E402
from src.backtest.returns import run_returns                 # noqa: E402
from src.backtest.stats import run_stats, sharpe_with_nw     # noqa: E402
from src.utils.seed import set_global_seed                   # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "phase2" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "phase2" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFIG = PROJECT_ROOT / "config" / "spy_phase2.yaml"

# ── attribution gate — pre-registered AND amended, both reported ────────────
# Pre-registered ("same as v1", tests/test_attribution_reconcile.py):
MAX_BOOK_RESIDUAL_ABS_SHARE = 0.20
MAX_POSITION_RESIDUAL_OVER_PREMIUM = 0.10   # a MAX, calibrated on v1's 10 positions
#
# Day-34 finding: the book-level bar passes on SPY (0.137) and the median
# position ratio (0.0175) beats v1's, but the per-position MAX fails (0.435):
# it is a max statistic over 294 draws instead of 10, on ~10%-IV straddles
# whose premiums are half as large relative to a daily move — and the window
# contains settlement bars and >2-sigma gap days (e.g. +2.07% at 1 DTE,
# 2024-02-22) where one-day Taylor is invalid by construction (the blow-up
# documented in attribution.py since Day 21). No scoping makes a max-of-294
# pass a bar calibrated on a max-of-10.
#
# AMENDED per-position bar (user-approved 2026-07-16, BEFORE portfolio/costs/
# returns/metrics existed; the pre-registered verdict is still computed and
# reported as FAILED alongside — nothing is buried):
#   p95 of residual/premium EXCLUDING the settlement bar  < 0.10
#     (quantile scales with n; the settlement bar is mark-to-intrinsic — exact,
#      model-free — so its Taylor error says nothing about the living book)
#   absolute worst position (settlement included)         < 0.50  (sanity cap)
#   book-level share (settlement included)                < 0.20  (unchanged)
MAX_P95_RESIDUAL_EX_SETTLEMENT = 0.10
MAX_POSITION_RESIDUAL_SANITY = 0.50

# pre-registered folds (config/spy_phase2.yaml design block)
BURN_IN_END = pd.Timestamp("2023-09-30")
FOLDS = {
    "2023Q4": ("2023-10-01", "2023-12-31"),
    "2024Q1": ("2024-01-01", "2024-03-31"),
    "2024Q2": ("2024-04-01", "2024-06-30"),
}


def fold_report(returns_path: Path, capital_base: float) -> dict:
    """Per-fold PnL / Sharpe / NW t from the daily net return series."""
    ret = pd.read_parquet(returns_path)
    ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.set_index("date")["daily_return"]

    def block(r: pd.Series) -> dict:
        arr = r.to_numpy()
        sh = sharpe_with_nw(arr, n_lags=int(np.ceil(np.sqrt(arr.size))))
        return {
            "n_days": int(arr.size),
            "net_pnl_usd": float(arr.sum() * capital_base),
            "sharpe_annualized": sh["sharpe_annualized"],
            "nw_tstat": sh["nw_tstat"],
            "worst_day_usd": float(arr.min() * capital_base),
            "best_day_usd": float(arr.max() * capital_base),
        }

    folds = {}
    for name, (s, e) in FOLDS.items():
        folds[name] = block(daily.loc[s:e])
    oos = daily.loc[pd.Timestamp(FOLDS["2023Q4"][0]):]
    burn = daily.loc[:BURN_IN_END]
    # positions entered inside the window run to expiry (last: 2024-08-16), so
    # PnL accrues past the last fold's end — the settlement tail. It contains
    # the 2024-08-05 vol spike, so hiding it inside an aggregate (or dropping
    # it) would each be a different lie; it gets its own labelled bucket and
    # all_test_folds = folds + tail, reconciled exactly.
    tail = daily.loc[pd.Timestamp(FOLDS["2024Q2"][1]) + pd.Timedelta(days=1):]

    return {
        "design": "walk-forward evaluation slices per config/spy_phase2.yaml; "
                  "nothing in the primary chain is refit on the window "
                  "(HAR is expanding-OOS by construction), so folds are "
                  "honest out-of-sample quarters, not refit boundaries",
        "capital_base_usd": capital_base,
        "burn_in": {"through": str(BURN_IN_END.date()), **block(burn)},
        "folds": folds,
        "settlement_tail": {
            "note": "expiry run-off of positions entered by 2024-06-28; "
                    "contains the 2024-08-05 vol spike",
            "from": str((pd.Timestamp(FOLDS["2024Q2"][1])
                         + pd.Timedelta(days=1)).date()),
            **block(tail)},
        "all_test_folds": block(oos),
        "positive_folds": sum(f["net_pnl_usd"] > 0 for f in folds.values()),
        "n_folds": len(folds),
    }


def main() -> None:
    set_global_seed()
    for f in ("signal.parquet", "svi_params_joint.parquet"):
        if not (PROCESSED_DIR / f).exists():
            raise FileNotFoundError(f"{PROCESSED_DIR / f} missing — run Days 32-33 first")

    price_path = load_price_path(RAW_DIR, "spy_ohlc.parquet", "spy_ohlc_ext.parquet")
    last_bar = price_path["date"].max()

    # ── Day-22 gate first: no PnL claim before attribution closes ──────────
    rec = run_reconcile(processed_dir=PROCESSED_DIR, price_path=price_path,
                        report_path=RESULTS_DIR / "attribution_reconcile_spy.json",
                        plots_dir=PLOTS_DIR, settlement_split=True)
    share = rec["book_residual_abs_share"]
    worst = rec["worst_position_residual_over_premium"]
    p95_ex = rec["p95_residual_over_premium_ex_settlement"]

    prereg_pass = (share < MAX_BOOK_RESIDUAL_ABS_SHARE
                   and worst < MAX_POSITION_RESIDUAL_OVER_PREMIUM)
    amended_pass = (share < MAX_BOOK_RESIDUAL_ABS_SHARE
                    and p95_ex < MAX_P95_RESIDUAL_EX_SETTLEMENT
                    and worst < MAX_POSITION_RESIDUAL_SANITY)

    gate = {
        "preregistered": {
            "pass": prereg_pass,
            "book_residual_abs_share": share,
            "worst_position_residual_over_premium": worst,
            "thresholds": {"book": MAX_BOOK_RESIDUAL_ABS_SHARE,
                           "worst_position": MAX_POSITION_RESIDUAL_OVER_PREMIUM},
            "note": "v1's per-position bar is a MAX calibrated on 10 positions;"
                    " see amended block for why it mechanically fails at n=294",
        },
        "amended": {
            "pass": amended_pass,
            "p95_residual_over_premium_ex_settlement": p95_ex,
            "median_residual_over_premium": rec["median_residual_over_premium"],
            "thresholds": {"book": MAX_BOOK_RESIDUAL_ABS_SHARE,
                           "p95_ex_settlement": MAX_P95_RESIDUAL_EX_SETTLEMENT,
                           "worst_position_sanity": MAX_POSITION_RESIDUAL_SANITY},
            "amended_when": "2026-07-16, before portfolio/costs/returns/metrics"
                            " existed for SPY; user-approved",
            "mechanism": "one-day Taylor breakdown on settlement bars and"
                         " >2-sigma gap days at <=3 DTE (attribution.py's"
                         " documented blow-up); median violator has 73% of"
                         " |residual| in the last 3 DTE",
        },
    }
    (RESULTS_DIR / "attribution_gate_spy.json").write_text(
        json.dumps(gate, indent=2), newline="\n")

    print(f"\nATTRIBUTION GATE (pre-registered): book {share:.3f} "
          f"(< {MAX_BOOK_RESIDUAL_ABS_SHARE}) | worst {worst:.3f} "
          f"(< {MAX_POSITION_RESIDUAL_OVER_PREMIUM}) -> "
          f"{'PASS' if prereg_pass else 'FAIL'}")
    print(f"ATTRIBUTION GATE (amended)       : p95 ex-settlement {p95_ex:.3f} "
          f"(< {MAX_P95_RESIDUAL_EX_SETTLEMENT}) | worst {worst:.3f} "
          f"(< {MAX_POSITION_RESIDUAL_SANITY}) -> "
          f"{'PASS' if amended_pass else 'FAIL'}")
    if not amended_pass:
        raise SystemExit("AMENDED ATTRIBUTION GATE FAILED on SPY — stopping "
                         "before any PnL is computed (v1 Day-22 rule).")
    print("proceeding under the AMENDED gate; the pre-registered verdict is "
          "reported alongside, not replaced\n")

    run_portfolio(processed_dir=PROCESSED_DIR, price_path=price_path,
                  summary_path=RESULTS_DIR / "portfolio_summary_spy.json",
                  plots_dir=PLOTS_DIR, config_path=CONFIG)
    run_costs(processed_dir=PROCESSED_DIR, price_path=price_path,
              summary_path=RESULTS_DIR / "costs_summary_spy.json",
              plots_dir=PLOTS_DIR, config_path=CONFIG)
    rsum = run_returns(processed_dir=PROCESSED_DIR, price_path=price_path,
                       summary_path=RESULTS_DIR / "returns_summary_spy.json",
                       plots_dir=PLOTS_DIR, config_path=CONFIG)

    # metrics/stats read returns_summary.json/costs_summary.json by fixed name
    # from results_dir — give them a phase2-local view of those names
    (RESULTS_DIR / "returns_summary.json").write_text(
        (RESULTS_DIR / "returns_summary_spy.json").read_text(), newline="\n")
    (RESULTS_DIR / "costs_summary.json").write_text(
        (RESULTS_DIR / "costs_summary_spy.json").read_text(), newline="\n")
    n_trades = rsum["n_positions"]
    run_metrics(processed_dir=PROCESSED_DIR, results_dir=RESULTS_DIR, notes={
        "headline": "Sharpe is NOT the headline for a short-vol book "
                    "(flatters negative skew); read skew/kurtosis/CVaR/"
                    "Calmar/Sortino as co-headlines.",
        "per_trade_sample": f"{n_trades} trades across 147 sessions — 29x "
                            "v1's sample; tail stats now have real support.",
        "settlement": f"price path through {last_bar.date()} "
                      "(spy_ohlc_ext) covers every traded expiry.",
    })
    run_stats(processed_dir=PROCESSED_DIR, results_dir=RESULTS_DIR,
              price_path=price_path, interpretation="auto")
    # metrics.json is the merged artifact; rename to the phase2 convention and
    # drop the fixed-name views so nothing pretends to be a v1 artifact
    (RESULTS_DIR / "metrics.json").replace(RESULTS_DIR / "metrics_spy.json")
    (RESULTS_DIR / "returns_summary.json").unlink()
    (RESULTS_DIR / "costs_summary.json").unlink()

    # ── walk-forward fold report ────────────────────────────────────────────
    wf = fold_report(PROCESSED_DIR / "returns.parquet", rsum["capital_base_usd"])
    wf_path = RESULTS_DIR / "walkforward_spy.json"
    wf_path.write_text(json.dumps(wf, indent=2), newline="\n")

    print(f"\nwalk-forward ({wf['n_folds']} folds, burn-in through "
          f"{wf['burn_in']['through']}):")
    for name, f in wf["folds"].items():
        print(f"  {name}: net ${f['net_pnl_usd']:+9.2f} | Sharpe "
              f"{f['sharpe_annualized']:+5.2f} | NW t {f['nw_tstat']:+5.2f} "
              f"| {f['n_days']} days")
    t = wf["settlement_tail"]
    print(f"  tail: net ${t['net_pnl_usd']:+9.2f} | Sharpe "
          f"{t['sharpe_annualized']:+5.2f} | NW t {t['nw_tstat']:+5.2f} "
          f"| {t['n_days']} days (expiry run-off from {t['from']})")
    a = wf["all_test_folds"]
    print(f"  ALL : net ${a['net_pnl_usd']:+9.2f} | Sharpe "
          f"{a['sharpe_annualized']:+5.2f} | NW t {a['nw_tstat']:+5.2f}")
    print(f"-> {wf_path}")


if __name__ == "__main__":
    main()
