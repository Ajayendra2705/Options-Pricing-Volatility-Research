"""
Day 21 — Greeks PnL attribution.

Per leg, per bar, Taylor-expand the mark around the PREVIOUS close
(state S_{t-1}, tau_{t-1}, sigma_{t-1}; all Greeks evaluated there):

    pnl ~ delta*dS + 0.5*gamma*dS^2 + theta*dt
        + vega*dsig + vanna*dS*dsig + 0.5*volga*dsig^2 + rho*dr + residual

Book level adds the two terms the engine ledger makes EXACT (no Taylor
error): hedge holding PnL shares_{t-1}*dS and cash financing
cash_{t-1}*(e^{r dt}-1). Rebalancing/settlement flows never touch equity
(self-financing), so

    book residual == sum of per-leg option Taylor errors, nothing else.

The engine marks each leg at a constant vol, so dsig = 0 and the three vol
terms vanish on engine output today — but the machinery takes a per-leg
sigma series so surface re-marking (and the Day-22 reconciliation on real
data) reuses it unchanged. r is a constant by construction in the engine,
so the rho term is identically zero until a rate series exists; the column
stays in the output because the decomposition is pre-registered in PLAN.

Residual is DEFINED as actual minus explained — attribution can't cheat.
The content is in how small it is (Day-22 gate); known blow-up: the last
bars before expiry, where gamma explodes and one-day Taylor breaks down.

NO LOOKAHEAD: the row for date t uses S_{t-1}, S_t and earlier only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import DAYS_PER_YEAR, Leg, _tau
from src.greeks.black_scholes import (delta_spot, gamma_spot, theta_spot,
                                      vanna_spot, vega_spot, volga_spot)

TERM_COLS = ["delta_pnl", "gamma_pnl", "theta_pnl",
             "vega_pnl", "vanna_pnl", "volga_pnl", "rho_pnl"]


def _sigma_at(sig, date) -> float:
    return float(sig.loc[date]) if isinstance(sig, pd.Series) else float(sig)


def leg_attribution(dates, S, legs: list[Leg], sigma_by_leg=None,
                    r: float = 0.0, q: float = 0.0) -> pd.DataFrame:
    """Per-leg Taylor terms for every bar t >= 1 where the leg was alive at
    t-1 (its expiry-day settlement bar included: the mark-to-intrinsic move
    is still the same one-day move the expansion approximates).

    sigma_by_leg: {leg_no: float | pd.Series indexed by date}; defaults to
    each leg's constant mark_vol (engine convention, dsig = 0).
    """
    dates = pd.DatetimeIndex(dates)
    S = np.asarray(S, float)
    if len(dates) != len(S):
        raise ValueError("dates and S misaligned")
    if sigma_by_leg is None:
        sigma_by_leg = {j: l.mark_vol for j, l in enumerate(legs)}

    rows = []
    for t in range(1, len(dates)):
        d0, d1 = dates[t - 1], dates[t]
        s0 = S[t - 1]
        ds = S[t] - s0
        dt = (d1 - d0).days / DAYS_PER_YEAR
        for j, leg in enumerate(legs):
            tau0 = _tau(d0, leg.expiry)
            if tau0 <= 0.0:                      # settled at or before t-1
                continue
            sig0 = _sigma_at(sigma_by_leg[j], d0)
            dsig = _sigma_at(sigma_by_leg[j], d1) - sig0
            sc = leg.qty * leg.mult
            a = (s0, leg.K, tau0, sig0, r, q)
            rows.append({
                "date": d1, "leg": j,
                "delta_pnl": sc * float(delta_spot(*a, leg.cp)) * ds,
                "gamma_pnl": 0.5 * sc * float(gamma_spot(*a)) * ds * ds,
                "theta_pnl": sc * float(theta_spot(*a, leg.cp)) * dt,
                "vega_pnl": sc * float(vega_spot(*a)) * dsig,
                "vanna_pnl": sc * float(vanna_spot(*a)) * ds * dsig,
                "volga_pnl": 0.5 * sc * float(volga_spot(*a)) * dsig * dsig,
                "rho_pnl": 0.0,                  # dr = 0: engine r constant
            })
    return pd.DataFrame(rows, columns=["date", "leg", *TERM_COLS])


def book_attribution(ledger: pd.DataFrame, legs: list[Leg], sigma_by_leg=None,
                     r: float = 0.0, q: float = 0.0) -> pd.DataFrame:
    """Daily book-level decomposition against the engine's actual equity.

    Columns: the 7 Taylor terms (summed over legs) + hedge_pnl + financing
    + explained + actual (equity diff) + residual (= actual - explained).
    First bar (entry) is all zeros by construction.
    """
    la = leg_attribution(ledger["date"], ledger["S"], legs, sigma_by_leg, r, q)
    led = ledger.set_index("date")
    terms = (la.groupby("date")[TERM_COLS].sum()
             .reindex(led.index, fill_value=0.0))
    hedge = (led["shares"].shift(1) * led["S"].diff()).fillna(0.0)
    # ledger cash is recorded pre-accrual; overnight interest lands next bar
    ddays = led.index.to_series().diff().dt.days
    fin = (led["cash"].shift(1)
           * (np.exp(r * ddays / DAYS_PER_YEAR) - 1.0)).fillna(0.0)
    out = terms.copy()
    out["hedge_pnl"] = hedge
    out["financing"] = fin
    out["explained"] = terms.sum(axis=1) + hedge + fin
    out["actual"] = led["equity"].diff().fillna(0.0)
    out["residual"] = out["actual"] - out["explained"]
    return out.reset_index()


def attribution_summary(book: pd.DataFrame) -> dict:
    """Reconciliation numbers the Day-22 gate will threshold."""
    live = book.iloc[1:]                          # skip the all-zero entry bar
    tot_abs = float(live["actual"].abs().sum())
    return {
        "term_totals": {c: float(book[c].sum())
                        for c in [*TERM_COLS, "hedge_pnl", "financing"]},
        "total_pnl": float(book["actual"].sum()),
        "residual_total": float(book["residual"].sum()),
        "residual_abs_sum_over_actual_abs_sum":
            float(live["residual"].abs().sum() / tot_abs) if tot_abs else 0.0,
        "residual_worst_bar": float(book["residual"].abs().max()),
    }


def demo(seed: int = 42):
    """Day 21 deliverable: full decomposition of the Day-19 demo position;
    stacked cumulative terms vs actual equity, residual in its own panel."""
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.backtest.engine import PLOTS_DIR, run_hedged

    rng = np.random.default_rng(seed)
    n = 43
    dates = pd.bdate_range("2023-06-02", periods=n)
    dt = 1.0 / 252
    S = 180.0 * np.exp(np.cumsum(
        np.r_[0.0, -0.5 * 0.20 ** 2 * dt + 0.20 * np.sqrt(dt)
              * rng.standard_normal(n - 1)]))
    legs = [Leg(K=180.0, expiry=dates[-1], cp=+1, qty=-1, mark_vol=0.25),
            Leg(K=180.0, expiry=dates[-1], cp=-1, qty=-1, mark_vol=0.25)]
    led = run_hedged(dates, S, legs, r=0.0)
    book = book_attribution(led, legs, r=0.0)
    print(json.dumps(attribution_summary(book), indent=2))

    show = ["theta_pnl", "gamma_pnl", "delta_pnl", "hedge_pnl"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  height_ratios=[2, 1])
    for c in show:
        ax.plot(book["date"], book[c].cumsum(), lw=1.1, label=c)
    ax.plot(book["date"], book["explained"].cumsum(), lw=1.0, ls="--",
            color="0.4", label="explained")
    ax.plot(book["date"], book["actual"].cumsum(), lw=1.8, color="k",
            label="actual equity")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_ylabel("cumulative PnL ($)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Greeks attribution: short 180 straddle (IV 25% / RV 20%)")
    ax2.bar(book["date"], book["residual"], width=0.8, color="firebrick")
    ax2.set_ylabel("residual/bar ($)")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOTS_DIR / "attribution_demo.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(f"-> {demo()}")
