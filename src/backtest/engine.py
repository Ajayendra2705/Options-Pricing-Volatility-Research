"""
Day 19 — Delta-hedge engine core.

Self-financing daily ledger for a book of European-style option legs hedged
with the underlying:

    equity_t = cash_t + shares_t * S_t + V_t(options)

- Marking: each leg carries its own constant mark vol (entry IV). Surface
  re-marking is a later refinement; constant-vol marking keeps the Day-20+
  Greeks attribution clean (no vol-move term until it's introduced
  deliberately).
- Hedging: at each hedge date (every `hedge_every` bars, always at entry),
  shares are set to -portfolio delta. Rebalancing trades cash for stock at
  the same price, so it NEVER changes equity — hedge PnL comes only from
  holding, which is what makes the ledger self-financing by construction.
- Settlement: on a leg's expiry date its intrinsic value moves to cash and
  the leg dies. When the whole book is dead the hedge is liquidated.
- Cash accrues at r (ACT/365, discrete daily compounding e^{r dt}).
- No transaction costs here — cost model arrives on Day 21 (config
  pre-registers it); the ledger records traded shares so costs bolt on
  without touching this file's accounting.

NO LOOKAHEAD: everything at row t uses S_t and earlier only. Verified by an
invariance test (corrupt future prices -> ledger through t unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.greeks.black_scholes import delta_spot, price_spot

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

DAYS_PER_YEAR = 365.0    # calendar-time tau for pricing, matches Part 1


@dataclass(frozen=True)
class Leg:
    """One option position. qty > 0 long, < 0 short (contracts)."""
    K: float
    expiry: pd.Timestamp
    cp: int                  # +1 call, -1 put
    qty: float
    mark_vol: float
    mult: float = 100.0


def _tau(date: pd.Timestamp, expiry: pd.Timestamp) -> float:
    return max((expiry - date).days, 0) / DAYS_PER_YEAR


def _leg_value_delta(leg: Leg, S: float, date: pd.Timestamp,
                     r: float, q: float) -> tuple[float, float]:
    """(value, delta) in dollars/shares for one live leg."""
    t = _tau(date, leg.expiry)
    if t <= 0.0:
        raise AssertionError("expired leg must be settled, not marked")
    scale = leg.qty * leg.mult
    v = scale * price_spot(S, leg.K, t, leg.mark_vol, r, q, leg.cp)
    d = scale * delta_spot(S, leg.K, t, leg.mark_vol, r, q, leg.cp)
    return float(v), float(d)


def run_hedged(dates, S, legs: list[Leg], r: float = 0.0, q: float = 0.0,
               hedge_every: int = 1) -> pd.DataFrame:
    """Daily self-financing ledger. equity == cumulative PnL (starts at 0).

    dates: ascending trading dates covering entry .. >= max expiry.
    S:     underlying closes aligned with dates.
    """
    dates = pd.DatetimeIndex(dates)
    S = np.asarray(S, float)
    if len(dates) != len(S):
        raise ValueError("dates and S misaligned")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be strictly increasing")
    if dates[-1] < max(l.expiry for l in legs):
        raise ValueError("path must reach the last expiry")
    if any(l.expiry not in dates for l in legs):
        raise ValueError("every expiry must be a path date (settlement bar)")

    alive = list(legs)
    cash = 0.0
    shares = 0.0
    rows = []
    for i, (d, s) in enumerate(zip(dates, S)):
        # settle legs expiring today: intrinsic to cash
        settled = [l for l in alive if l.expiry == d]
        for l in settled:
            cash += l.qty * l.mult * max(l.cp * (s - l.K), 0.0)
        alive = [l for l in alive if l.expiry != d]

        # mark the book
        v_opt = d_opt = 0.0
        for l in alive:
            v, dd = _leg_value_delta(l, s, d, r, q)
            v_opt += v
            d_opt += dd

        if i == 0:
            cash -= v_opt                       # entry: premium funds the book

        traded = 0.0
        if not alive:
            traded = -shares                    # book dead: liquidate hedge
        elif i % hedge_every == 0:
            traded = -d_opt - shares            # target: flat total delta
        cash -= traded * s                      # trade at today's close
        shares += traded

        equity = cash + shares * s + v_opt      # unchanged by `traded` (self-financing)
        rows.append({"date": d, "S": s, "V_opt": v_opt, "delta_opt": d_opt,
                     "shares": shares, "traded": traded, "cash": cash,
                     "equity": equity})

        if i + 1 < len(dates):                  # overnight financing
            cash *= np.exp(r * (dates[i + 1] - d).days / DAYS_PER_YEAR)

    return pd.DataFrame(rows)


def plot_pnl_path(ledger: pd.DataFrame, title: str,
                  out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  height_ratios=[2, 1])
    ax.plot(ledger["date"], ledger["equity"], lw=1.5)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("hedged PnL ($)")
    ax.set_title(title)
    ax2.plot(ledger["date"], ledger["S"], color="0.4", lw=1.2)
    ax2.set_ylabel("underlying")
    p = Path(out_path)
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def demo(seed: int = 42) -> Path:
    """Day 19 deliverable: one short ATM straddle, delta-hedged daily, on a
    GBM path whose realized vol (20%) is below the mark vol (25%) — the
    hedged PnL should grind positive; sanity plot to results/plots/."""
    rng = np.random.default_rng(seed)
    n = 43                                       # ~2 months of bars
    dates = pd.bdate_range("2023-06-02", periods=n)
    dt = 1.0 / 252
    S = 180.0 * np.exp(np.cumsum(
        np.r_[0.0, -0.5 * 0.20 ** 2 * dt + 0.20 * np.sqrt(dt)
              * rng.standard_normal(n - 1)]))
    legs = [Leg(K=180.0, expiry=dates[-1], cp=+1, qty=-1, mark_vol=0.25),
            Leg(K=180.0, expiry=dates[-1], cp=-1, qty=-1, mark_vol=0.25)]
    led = run_hedged(dates, S, legs, r=0.0)
    print(f"demo: short 180 straddle, IV 25% vs RV 20%, "
          f"final PnL ${led['equity'].iloc[-1]:.0f}")
    return plot_pnl_path(
        led, "Short ATM straddle, daily delta hedge (IV 25% / RV 20%)",
        PLOTS_DIR / "engine_demo.png")


if __name__ == "__main__":
    print(f"-> {demo()}")
