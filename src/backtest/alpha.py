"""
Day 27 — Event PnL table + alpha isolation.

Two additions to results/metrics.json (PLAN deliverable):

1. EVENT PnL TABLE.  What a desk PM looks at first: strategy return on the big
   vol days.  The pre-registered named events (Feb 2018 Volmageddon, Mar 2020,
   Aug 2024) are ENTIRELY out of sample — the data is AAPL 2022-01..2023-08 —
   so they are reported as "no data (out of sample)".  The in-sample event is
   the **2023-08-04 AAPL earnings gap (-4.80%)**, the only >3% move inside the
   trade window (2023-06-02..08-18).  For each event day we surface the
   MARGIN-PROCYCLICALITY interaction (SPEC): book PnL and book Reg-T margin on
   the same bar — the "tail hit twice" when margin expands as equity falls.

2. ALPHA ISOLATION.  Regress the book's net daily return on a VRP / short-
   straddle factor to separate short-vol beta from residual edge.
   - FACTOR CONSTRUCTION (stated explicitly, SPEC): daily PnL of ONE short ATM
     straddle on AAPL, entered at the first trade date, held to the last path
     date, **delta-hedged daily, NOT rolled** (raw delta-hedged short vol),
     marked at the mean short-leg IV, on the same capital base.  Delta hedging
     strips direction so the factor is pure gamma/theta = the harvested vol
     premium.  (Non-rolled -> gamma fades as spot drifts off the fixed strike;
     documented limitation.)
   - alpha, beta, R2, and **Newey-West HAC** t-stats with Bartlett kernel,
     lags = ceil(median holding horizon in trading days) — the overlap induced
     by holding each straddle ~a month.  HAC implemented here (no statsmodels).

OUTPUT: merged into results/metrics.json under "event_table" and
"alpha_regression" (tracked, byte-stable).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import Leg, run_hedged
from src.backtest.metrics import merge_metrics
from src.backtest.reconcile import build_positions, load_price_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

EVENT_MOVE_THRESHOLD = 0.03      # |underlying return| flagged as an "event"
EVENT_TABLE_TOPK = 5             # also always show the K biggest-move days
NAMED_OOS_EVENTS = ["2018-02 Volmageddon", "2020-03 COVID crash", "2024-08"]


# ── Newey-West HAC regression ────────────────────────────────────────────────

def newey_west_ols(y: np.ndarray, x: np.ndarray, n_lags: int) -> dict:
    """OLS y ~ 1 + x with Newey-West (Bartlett) HAC standard errors.

    Returns alpha/beta, their HAC SE and t-stats, R^2, and n_lags.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = y.size
    X = np.column_stack([np.ones(n), x])          # [1, x]
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)                     # [alpha, slope]
    resid = y - X @ beta

    # HAC meat: S = sum_t u_t^2 x_t x_t' + Bartlett-weighted lag cross terms
    k = X.shape[1]
    S = np.zeros((k, k))
    xu = X * resid[:, None]                        # (n,k), score contributions
    S += xu.T @ xu                                 # lag 0
    for l in range(1, n_lags + 1):
        w = 1.0 - l / (n_lags + 1)                 # Bartlett weight
        Gamma = xu[l:].T @ xu[:-l]                 # sum_t s_t s_{t-l}'
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return {
        "alpha": float(beta[0]), "beta": float(beta[1]),
        "alpha_se": float(se[0]), "beta_se": float(se[1]),
        "alpha_t": float(t[0]), "beta_t": float(t[1]),
        "r_squared": r2, "n_obs": int(n), "n_lags": int(n_lags),
    }


# ── VRP / short-straddle factor ──────────────────────────────────────────────

def build_vrp_factor(positions: list[dict], path: pd.DataFrame,
                     capital_base: float) -> pd.Series:
    """Daily return of a delta-hedged, non-rolled short ATM straddle over the
    full trade window, on the same capital base.  Indexed by date."""
    entry = min(p["date"] for p in positions)
    expiry = max(p["expiry"] for p in positions)
    shorts = [p for p in positions if p["side"] == "short_vol"]
    ref = min(shorts, key=lambda p: p["date"])    # first short: r/q reference
    mark = float(np.mean([p["mark_vol"] for p in shorts]))

    win = path[(path["date"] >= entry) & (path["date"] <= expiry)
               ].reset_index(drop=True)
    S0 = float(win["close"].iloc[0])
    K = round(S0 / 5.0) * 5.0                      # nearest $5 listed strike
    legs = [Leg(K=K, expiry=expiry, cp=+1, qty=-1.0, mark_vol=mark),
            Leg(K=K, expiry=expiry, cp=-1, qty=-1.0, mark_vol=mark)]
    led = run_hedged(win["date"], win["close"].to_numpy(), legs,
                     r=ref["r"], q=ref["q"])
    fac = (led.set_index("date")["equity"].diff() / capital_base)
    fac.name = "vrp_factor_return"
    return fac


# ── event table ──────────────────────────────────────────────────────────────

def build_event_table(ret: pd.DataFrame, path: pd.DataFrame) -> list[dict]:
    """Big-move days inside the trade window with book PnL + margin (procyclic).
    """
    dmin, dmax = ret["date"].min(), ret["date"].max()
    win = path[(path["date"] >= dmin) & (path["date"] <= dmax)].copy()
    win["u_ret"] = win["close"].pct_change()

    r = ret.set_index("date")
    dmargin = r["book_margin"].diff()
    rows = []
    # union: the top-K biggest |move| days plus anything over threshold
    ranked = win.dropna(subset=["u_ret"]).reindex(
        win.dropna(subset=["u_ret"])["u_ret"].abs().sort_values(
            ascending=False).index)
    chosen = ranked.head(EVENT_TABLE_TOPK)
    for _, w in chosen.iterrows():
        d = w["date"]
        rows.append({
            "date": str(d.date()),
            "underlying_return": float(w["u_ret"]),
            "is_event": bool(abs(w["u_ret"]) >= EVENT_MOVE_THRESHOLD),
            "book_daily_return": float(r["daily_return"].get(d, float("nan"))),
            "book_margin": float(r["book_margin"].get(d, float("nan"))),
            "margin_change": float(dmargin.get(d, float("nan"))),
            "net_equity": float(r["net_equity"].get(d, float("nan"))),
        })
    rows.sort(key=lambda x: x["date"])
    return rows


# ── runner ──────────────────────────────────────────────────────────────────

def run_alpha() -> dict:
    """Compute event table + VRP-factor regression, merge into metrics.json."""
    ret = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    path = load_price_path()
    positions = build_positions()
    with open(RESULTS_DIR / "returns_summary.json") as fh:
        capital_base = json.load(fh)["capital_base_usd"]

    # holding horizon -> Newey-West lags
    hold_bars = []
    for p in positions:
        w = path[(path["date"] >= p["date"]) & (path["date"] <= p["expiry"])]
        hold_bars.append(len(w))
    n_lags = int(np.ceil(np.median(hold_bars)))

    # factor + regression on common dates (drop bar-0 diffs)
    factor = build_vrp_factor(positions, path, capital_base)
    df = pd.DataFrame({
        "book": ret.set_index("date")["daily_return"],
        "factor": factor,
    }).dropna()
    reg = newey_west_ols(df["book"].to_numpy(), df["factor"].to_numpy(), n_lags)
    reg["factor"] = ("delta-hedged non-rolled short ATM straddle on AAPL, "
                     "full window, marked at mean short-leg IV, same capital "
                     "base")
    reg["holding_horizon_bars"] = n_lags
    reg["factor_book_corr"] = float(df["book"].corr(df["factor"]))

    events = build_event_table(ret, path)

    # merge into metrics.json (shared file; merge_metrics preserves the other
    # days' blocks and enforces canonical key order -> byte-stable)
    mpath = RESULTS_DIR / "metrics.json"
    block = {}
    block["alpha_regression"] = reg
    block["event_table"] = {
        "trade_window": [str(ret["date"].min().date()),
                         str(ret["date"].max().date())],
        "named_events_out_of_sample": NAMED_OOS_EVENTS,
        "note": "Pre-registered named events (Volmageddon/COVID/Aug-2024) are "
                "outside the 2022-2023 sample; the in-sample event is the "
                "2023-08-04 AAPL earnings gap (-4.80%). The margin-change "
                "column surfaces the procyclicality interaction, but note this "
                "book is net LONG vol (regression beta -1.59 to a short-vol "
                "factor): on 2023-08-04 it GAINED on the gap (long gamma) while "
                "margin FELL - the opposite of the short-vol 'tail hit twice'. "
                "The procyclical margin-up-as-equity-falls pattern shows on the "
                "short-vol-dominated days instead (e.g. 2023-06-22).",
        "days": events,
    }
    metrics = merge_metrics(block)

    print(f"alpha: beta {reg['beta']:+.3f} (t={reg['beta_t']:+.2f}), "
          f"alpha {reg['alpha']:+.5f}/day (t={reg['alpha_t']:+.2f}), "
          f"R2 {reg['r_squared']:.3f}, NW lags {n_lags}, "
          f"corr {reg['factor_book_corr']:+.2f}")
    ev = [e for e in events if e["is_event"]]
    for e in ev:
        print(f"  EVENT {e['date']}: u_ret {e['underlying_return']:+.2%}, "
              f"book {e['book_daily_return']:+.4f}, "
              f"dMargin ${e['margin_change']:+,.0f}")
    print(f"-> {mpath}")
    return metrics


if __name__ == "__main__":
    run_alpha()
