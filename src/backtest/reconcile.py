"""
Day 22 — Attribution reconciliation on REAL data (the make-or-break gate).

Builds the pre-registered book (config/primary.yaml): one ATM straddle per
non-flat signal row — strike = nearest listed strike to the slice forward,
marked at the joint arb-free SVI vol at that strike, qty -1 (short_vol) or
+1 (long_vol) — runs the delta-hedge engine on the real AAPL close path to
each expiry, attributes every position with the Day-21 Greeks decomposition,
and gates the residual.

Rates: each slice carries its own market-implied discount rate r from the
Day-7 forward extraction; the carry q is backed out of the traded forward,
q = r - ln(F/S_entry)/T, so the engine's internal forward matches the
market forward at entry (dividends handled implicitly, no external feed).

Hedge path: entry..expiry closes from data/raw/aapl_ohlc.parquet extended
by aapl_ohlc_ext.parquet (2023-07..08, downloaded for settlement coverage
only). The extension is intentionally a SEPARATE raw file: signal/HAR/RV
inputs end at 2023-06-30, so the no-lookahead boundary between "data the
signal saw" and "path the trade lives on" is physical, and the original
raw file's hash is untouched.

Gate (tests/test_attribution_reconcile.py): the residual of the Day-21
decomposition — pure per-leg one-day Taylor error by construction — must
stay a small fraction of gross PnL flow, book-wide and per position.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.attribution import book_attribution
from src.backtest.engine import Leg, run_hedged
from src.surface.svi import svi_iv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SVI_COLS = ["a", "b", "rho", "m", "sigma"]


def load_price_path() -> pd.DataFrame:
    """Signal-window OHLC + post-June extension, close-only, deduped."""
    base = pd.read_parquet(RAW_DIR / "aapl_ohlc.parquet")[["date", "close"]]
    ext_p = RAW_DIR / "aapl_ohlc_ext.parquet"
    if ext_p.exists():
        ext = pd.read_parquet(ext_p)[["date", "close"]]
        base = pd.concat([base, ext])
    return (base.drop_duplicates("date").sort_values("date")
            .reset_index(drop=True))


def build_positions(signal: pd.DataFrame | None = None) -> list[dict]:
    """One straddle per non-flat pre-registered signal row.

    Returns dicts: date, expiry, side, qty, K, F, S0, r, q, mark_vol, T.
    """
    if signal is None:
        signal = pd.read_parquet(PROCESSED_DIR / "signal.parquet")
    fwd = pd.read_parquet(PROCESSED_DIR / "forwards.parquet")
    svi = pd.read_parquet(PROCESSED_DIR / "svi_params_joint.parquet")
    chain = pd.read_parquet(PROCESSED_DIR / "chain_clean.parquet")
    path = load_price_path().set_index("date")["close"]

    out = []
    for _, row in signal.iterrows():
        if row["side"] == "flat":
            continue
        key = (fwd["date"] == row["date"]) & (fwd["expiry"] == row["expiry"])
        f = fwd[key].iloc[0]
        p = svi[(svi["date"] == row["date"])
                & (svi["expiry"] == row["expiry"])].iloc[0]
        if not p["fit_ok"]:
            raise RuntimeError(f"unfitted slice traded: {row['date']}")
        sl = chain[(chain["date"] == row["date"])
                   & (chain["expiry"] == row["expiry"])]
        # straddle needs both sides quoted at the strike
        both = (sl.groupby("strike")["option_type"].nunique() == 2)
        strikes = both[both].index.to_numpy(float)
        K = float(strikes[np.argmin(np.abs(strikes - f["F"]))])

        S0 = float(path.loc[row["date"]])
        T = float(f["T"])
        r = float(f["r"])
        q = r - np.log(f["F"] / S0) / T          # carry implied by the forward
        mark = float(svi_iv(np.log(K / f["F"]), T, p[SVI_COLS].to_numpy(float)))
        out.append({
            "date": row["date"], "expiry": row["expiry"], "side": row["side"],
            "qty": -1.0 if row["side"] == "short_vol" else +1.0,
            "K": K, "F": float(f["F"]), "S0": S0, "r": r, "q": float(q),
            "mark_vol": mark, "T": T,
        })
    return out


def run_position(pos: dict, path: pd.DataFrame):
    """Engine + attribution for one straddle. Returns (ledger, book, legs)."""
    win = path[(path["date"] >= pos["date"])
               & (path["date"] <= pos["expiry"])]
    legs = [Leg(K=pos["K"], expiry=pos["expiry"], cp=+1, qty=pos["qty"],
                mark_vol=pos["mark_vol"]),
            Leg(K=pos["K"], expiry=pos["expiry"], cp=-1, qty=pos["qty"],
                mark_vol=pos["mark_vol"])]
    led = run_hedged(win["date"], win["close"].to_numpy(), legs,
                     r=pos["r"], q=pos["q"])
    book = book_attribution(led, legs, r=pos["r"], q=pos["q"])
    return led, book, legs


def run_reconcile() -> dict:
    """Full real-data reconciliation -> results/attribution_reconcile.json
    + residual plot. Returns the report dict (the Day-22 gate numbers)."""
    path = load_price_path()
    positions = build_positions()

    reports, resid_bars = [], []
    for pos in positions:
        led, book, _ = run_position(pos, path)
        live = book.iloc[1:]                      # entry bar is all zeros
        prem = float(abs(led["V_opt"].iloc[0]))
        reports.append({
            "date": str(pos["date"].date()),
            "expiry": str(pos["expiry"].date()),
            "side": pos["side"], "K": pos["K"],
            "mark_vol": pos["mark_vol"], "r": pos["r"], "q": pos["q"],
            "n_bars": int(len(led)),
            "premium": prem,
            "pnl": float(led["equity"].iloc[-1]),
            "theta_pnl": float(book["theta_pnl"].sum()),
            "gamma_pnl": float(book["gamma_pnl"].sum()),
            "residual_sum": float(book["residual"].sum()),
            "residual_abs_sum": float(live["residual"].abs().sum()),
            "actual_abs_sum": float(live["actual"].abs().sum()),
            "residual_over_premium": float(abs(book["residual"].sum()) / prem),
        })
        resid_bars.append(live["residual"])

    resid = pd.concat(resid_bars)
    tot_abs_resid = float(sum(r["residual_abs_sum"] for r in reports))
    tot_abs_actual = float(sum(r["actual_abs_sum"] for r in reports))
    report = {
        "n_positions": len(reports),
        "positions": reports,
        "book_residual_abs_share": tot_abs_resid / tot_abs_actual,
        "worst_position_residual_over_premium":
            max(r["residual_over_premium"] for r in reports),
        "total_pnl": float(sum(r["pnl"] for r in reports)),
        "total_residual": float(sum(r["residual_sum"] for r in reports)),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "attribution_reconcile.json", "w") as fh:
        json.dump(report, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax.hist(resid, bins=40, color="steelblue", edgecolor="0.3")
    ax.set_xlabel("per-bar residual ($)")
    ax.set_title(f"Attribution residuals, all positions "
                 f"(abs share {report['book_residual_abs_share']:.1%})")
    labels = [f"{r['date']}\n{r['expiry']}" for r in reports]
    ax2.bar(range(len(reports)),
            [r["residual_sum"] for r in reports], color="firebrick")
    ax2.set_xticks(range(len(reports)), labels, fontsize=6, rotation=45)
    ax2.set_title("cumulative residual per position ($)")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "attribution_residuals.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    return report


if __name__ == "__main__":
    rep = run_reconcile()
    print(json.dumps({k: v for k, v in rep.items() if k != "positions"},
                     indent=2))
    for p in rep["positions"]:
        print(f"{p['date']} {p['expiry']} {p['side']:>9} K={p['K']:<6} "
              f"pnl={p['pnl']:>8.2f} resid={p['residual_sum']:>7.2f} "
              f"({p['residual_over_premium']:.1%} of premium)")
