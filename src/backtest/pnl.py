"""
Day 20 — Per-leg PnL analytics: dollar-gamma-weighted realized vol and
break-even vol per position.

THE TWO RV OBJECTS (SPEC): the Yang-Zhang/HAR RV in the signal is a property
of the underlying alone. The RV that drives a delta-hedged option's PnL is
weighted by the position's OWN dollar gamma along its OWN path:

    sigma^2_gw = sum_t |$G_{t-1}| * ret_t^2  /  sum_t |$G_{t-1}| * dt_t

with $G = qty*mult*gamma*S^2 (position dollar gamma, previous close),
ret_t = ln(S_t/S_{t-1}), dt_t = calendar ACT/365 between bars (same clock the
marks accrue theta on). sigma_gw is the position's BREAK-EVEN vol: to first
order the hedged PnL of a constant-vol-marked book is

    pnl ~ sum_t 0.5 * $G_{t-1} * (ret_t^2 - sigma_mark^2 * dt_t)

so pnl >= 0 for a short-gamma book exactly when sigma_gw <= sigma_mark (and
vice versa for long). `theta_gamma_pnl` computes that series per leg; the
engine's actual equity must track its cumulative sum (verified in tests) —
this identity is the seed of the Day-23 attribution, where the residual gets
pinned down instead of waved at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def _bars(ledger: pd.DataFrame) -> pd.DataFrame:
    d = ledger[["date", "S"]].copy()
    d["ret"] = np.log(d["S"] / d["S"].shift(1))
    d["dt"] = d["date"].diff().dt.days / DAYS_PER_YEAR
    return d


def gamma_weighted_rv(ledger: pd.DataFrame, leg_ledger: pd.DataFrame,
                      leg: int | None = None) -> float:
    """Annualized dollar-gamma-weighted realized vol (position or one leg)."""
    gl = leg_ledger if leg is None else leg_ledger[leg_ledger["leg"] == leg]
    dg = gl.groupby("date")["dollar_gamma"].sum()
    bars = _bars(ledger).set_index("date")
    w = dg.reindex(bars.index).shift(1).abs()          # $gamma at prior close
    num = float((w * bars["ret"] ** 2).sum())
    den = float((w * bars["dt"]).sum())
    if den <= 0.0:
        raise ValueError("no dollar gamma held over the window")
    return float(np.sqrt(num / den))


def theta_gamma_pnl(ledger: pd.DataFrame, leg_ledger: pd.DataFrame,
                    mark_vols: dict[int, float]) -> pd.Series:
    """First-order hedged PnL per bar: 0.5*sum_legs $G_(t-1)*(ret^2 - sig^2 dt)."""
    bars = _bars(ledger).set_index("date")
    out = pd.Series(0.0, index=bars.index)
    for leg, sig in mark_vols.items():
        dg = (leg_ledger[leg_ledger["leg"] == leg]
              .set_index("date")["dollar_gamma"]
              .reindex(bars.index, fill_value=0.0).shift(1))
        out = out + 0.5 * dg * (bars["ret"] ** 2 - sig ** 2 * bars["dt"])
    return out.fillna(0.0).rename("theta_gamma_pnl")


def breakeven_report(ledger: pd.DataFrame, leg_ledger: pd.DataFrame,
                     mark_vols: dict[int, float]) -> dict:
    """Per-position break-even documentation (PLAN Day 20 deliverable)."""
    sigma_gw = gamma_weighted_rv(ledger, leg_ledger)
    # book-level mark: dollar-gamma-weighted average of leg marks
    w = leg_ledger.groupby("leg")["dollar_gamma"].apply(lambda s: s.abs().sum())
    sigma_mark = float(sum(w[l] * v for l, v in mark_vols.items()) / w.sum())
    pnl = float(ledger["equity"].iloc[-1])
    net_gamma_sign = float(np.sign(leg_ledger["dollar_gamma"].sum()))
    return {
        "sigma_mark_book": sigma_mark,
        "sigma_gamma_weighted": sigma_gw,
        "breakeven_gap_volpts": (sigma_gw - sigma_mark) * 100,
        "net_gamma_sign": net_gamma_sign,
        "pnl": pnl,
        "pnl_consistent_with_gap":
            bool(pnl * net_gamma_sign * (sigma_gw - sigma_mark) >= 0),
        "per_leg_pnl": {
            int(l): float(g["pnl_day"].sum())
            for l, g in leg_ledger.groupby("leg")},
    }


def demo(seed: int = 42):
    """Day 20 deliverable: per-leg ledger + break-even documentation for the
    Day-19 demo position; plot equity vs its theta-gamma first-order twin."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.backtest.engine import PLOTS_DIR, Leg, run_hedged

    rng = np.random.default_rng(seed)
    n = 43
    dates = pd.bdate_range("2023-06-02", periods=n)
    dt = 1.0 / 252
    S = 180.0 * np.exp(np.cumsum(
        np.r_[0.0, -0.5 * 0.20 ** 2 * dt + 0.20 * np.sqrt(dt)
              * rng.standard_normal(n - 1)]))
    legs = [Leg(K=180.0, expiry=dates[-1], cp=+1, qty=-1, mark_vol=0.25),
            Leg(K=180.0, expiry=dates[-1], cp=-1, qty=-1, mark_vol=0.25)]
    led, legs_led = run_hedged(dates, S, legs, r=0.0, return_legs=True)
    marks = {j: l.mark_vol for j, l in enumerate(legs)}
    rep = breakeven_report(led, legs_led, marks)
    print(json.dumps(rep, indent=2))

    tg = theta_gamma_pnl(led, legs_led, marks).cumsum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(led["date"], led["equity"], lw=1.5, label="engine equity")
    ax.plot(led["date"], tg.reindex(pd.Index(led["date"])).to_numpy(), "--",
            lw=1.3, label="first-order 0.5*$Γ(r² − σ²dt) cumulative")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("PnL ($)")
    ax.legend()
    ax.set_title(f"Short straddle: PnL vs theta-gamma twin "
                 f"(σ_gw {rep['sigma_gamma_weighted']:.1%} vs mark 25%)")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / "pnl_decomposition.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(f"-> {demo()}")
