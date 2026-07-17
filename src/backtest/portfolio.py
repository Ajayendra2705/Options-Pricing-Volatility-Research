"""
Day 23 — Portfolio assembly + sizing.

Takes the Day-22 pre-registered positions (one ATM straddle per non-flat
signal row, unit qty) and:

  1. Sizes each by vega + gamma (binding constraint wins) to fit per-position
     risk limits.
  2. Clips the sized book to portfolio-level inventory limits (gross vega,
     gross gamma) with pro-rata scaling.
  3. Runs the Day-19 delta-hedge engine on the real AAPL close path for each
     sized position.
  4. Aggregates per-position equity curves into a single portfolio PnL series
     (equity_t = Σ equity_i_t for all positions live at t).
  5. Applies a drawdown kill-switch: if peak-to-trough exceeds a threshold,
     all positions freeze (equity stays at the kill level).

OUTPUT:
  - data/processed/portfolio.parquet — daily portfolio series
  - results/portfolio_summary.json — sizing table + stats
  - results/plots/portfolio_equity.png — equity + DD subplot

SIZING IS RISK-BASED, NOT SIGNAL-PROPORTIONAL.  With a 3-slice cross-section
per date and 5 quote dates, proportional sizing would be noise-fitted.  Every
non-flat position gets the same risk budget, clipped by per-position and
portfolio limits (SPEC: "size by vega + gamma under inventory limits").

EQUITY = CUMULATIVE PnL.  The Day-19 engine starts equity at 0 (self-financing).
Capital-base / margin-denominated returns arrive on Day 25.

NO LOOKAHEAD: sizing uses only Greeks at entry (S_entry, K, T_entry, σ_mark,
r, q), all known at the close of the quote date.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.engine import Leg, run_hedged
from src.backtest.reconcile import build_positions, load_price_path
from src.greeks.black_scholes import gamma_spot, vega_spot

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFIG_PATH = PROJECT_ROOT / "config" / "primary.yaml"

MULT = 100.0  # contract multiplier (matches Leg default)


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_sizing_params(config_path: Path | None = None) -> dict:
    """Load sizing parameters from primary.yaml, with safe defaults.

    The pre-registered primary.yaml contains informal prose with unquoted
    colons (e.g. 'SPEC: margin-based returns') which makes the whole file
    unparseable by strict YAML.  We extract only the 'sizing:' block
    and parse it in isolation.  `config_path` lets Phase 2 read its own
    pre-registration (spy_phase2.yaml) instead of v1's.
    """
    config_path = config_path or CONFIG_PATH
    defaults = {
        "vega_limit_per_position": 500.0,
        "gamma_limit_per_position": 5000.0,
        "portfolio_gross_vega_limit": 3000.0,
        "portfolio_gross_gamma_limit": 30000.0,
        "max_drawdown_pct": 0.15,
    }
    if config_path.exists():
        try:
            lines = config_path.read_text().splitlines()
            # Find the sizing block and extract it
            sizing_lines = []
            in_sizing = False
            for line in lines:
                if line.strip().startswith("sizing:"):
                    in_sizing = True
                    sizing_lines.append(line)
                    continue
                if in_sizing:
                    # Stop at next top-level key or blank line after content
                    if line and not line[0].isspace() and not line.startswith("#"):
                        break
                    sizing_lines.append(line)
            if sizing_lines:
                cfg = yaml.safe_load("\n".join(sizing_lines))
                if cfg and "sizing" in cfg:
                    defaults.update(cfg["sizing"])
        except Exception:
            pass  # fall back to defaults
    return defaults


def _entry_greeks(pos: dict) -> tuple[float, float]:
    """Vega and dollar-gamma at entry for a unit straddle (call + put).

    Returns (|vega_notional|, |dollar_gamma|) for qty=±1.
    """
    S, K, T, sig = pos["S0"], pos["K"], pos["T"], pos["mark_vol"]
    r, q = pos["r"], pos["q"]
    # Straddle = call + put; vega is cp-independent, gamma is cp-independent
    v = float(vega_spot(S, K, T, sig, r, q))  # per-share, one side
    g = float(gamma_spot(S, K, T, sig, r, q))
    vega_straddle = 2.0 * abs(v) * MULT   # call vega + put vega (equal)
    dgamma_straddle = 2.0 * abs(g) * S * S * MULT   # call + put dollar gamma
    return vega_straddle, dgamma_straddle


# ── sizing ───────────────────────────────────────────────────────────────────

def size_position(pos: dict, params: dict) -> float:
    """Scale unit qty so neither vega nor gamma limit is breached.

    Returns the sized qty (preserving the sign from pos["qty"]).
    """
    vega_1, dgamma_1 = _entry_greeks(pos)
    sign = np.sign(pos["qty"])

    # Scale to each limit individually
    if vega_1 > 0:
        qty_vega = params["vega_limit_per_position"] / vega_1
    else:
        qty_vega = 1.0  # degenerate: no vega at all → pass through
    if dgamma_1 > 0:
        qty_gamma = params["gamma_limit_per_position"] / dgamma_1
    else:
        qty_gamma = 1.0  # degenerate: no gamma at all → pass through

    qty_abs = min(qty_vega, qty_gamma)
    return float(sign * qty_abs)


def portfolio_limits(sized_positions: list[dict],
                     params: dict) -> list[dict]:
    """Pro-rata clip to portfolio-level gross vega and gross gamma limits.

    Modifies and returns the same list (in-place qty updates).
    """
    # Compute current gross exposures
    gross_vega = sum(abs(p["sized_vega"]) for p in sized_positions)
    gross_gamma = sum(abs(p["sized_dgamma"]) for p in sized_positions)

    vega_ratio = (params["portfolio_gross_vega_limit"] / gross_vega
                  if gross_vega > 0 else 1.0)
    gamma_ratio = (params["portfolio_gross_gamma_limit"] / gross_gamma
                   if gross_gamma > 0 else 1.0)
    clip = min(vega_ratio, gamma_ratio, 1.0)  # only clip down, never up

    if clip < 1.0:
        for p in sized_positions:
            p["sized_qty"] *= clip
            p["sized_vega"] *= clip
            p["sized_dgamma"] *= clip
    return sized_positions


# ── drawdown kill-switch ─────────────────────────────────────────────────────

def drawdown_kill(equity: pd.Series, max_dd_pct: float
                  ) -> tuple[pd.Series, str | None]:
    """If peak-to-trough exceeds max_dd_pct of the peak equity, freeze
    equity at the kill level.

    Returns (modified equity series, kill_date or None).

    Peak equity = running max of cumulative PnL.  Since equity starts at 0,
    the peak can be 0 in the early bars — DD is then defined as the absolute
    loss, which is compared to the premium-at-risk (not the peak).  To avoid
    a division-by-zero on zero peak, we use an absolute drawdown threshold
    derived from the first non-zero peak: dd_abs > max_dd_pct * peak, where
    peak = max(equity_cummax, 1.0) to anchor the check.
    """
    eq = equity.copy()
    peak = eq.cummax()
    kill_date = None
    for i in range(len(eq)):
        ref = max(peak.iloc[i], 1.0)  # anchor: at least $1 avoids /0
        dd = peak.iloc[i] - eq.iloc[i]
        if dd > max_dd_pct * ref:
            kill_date = str(eq.index[i])
            eq.iloc[i:] = eq.iloc[i]
            break
    return eq, kill_date


# ── portfolio runner ─────────────────────────────────────────────────────────

def run_portfolio(
    params: dict | None = None,
    processed_dir: Path | None = None,
    price_path: pd.DataFrame | None = None,
    summary_path: Path | None = None,
    plots_dir: Path | None = None,
    make_plots: bool = True,
    config_path: Path | None = None,
) -> dict:
    """Full Day-23 deliverable: sized positions → portfolio equity curve.

    Returns the summary dict (also written to JSON). Seams default to v1's
    constants (Day-32 convention) so v1's paths never move.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    summary_path = summary_path or RESULTS_DIR / "portfolio_summary.json"
    plots_dir = plots_dir or PLOTS_DIR
    if params is None:
        params = _load_sizing_params(config_path)

    path = load_price_path() if price_path is None else price_path
    positions = build_positions(processed_dir=processed_dir, price_path=path)

    # ── 1. size each position ────────────────────────────────────────────
    for pos in positions:
        pos["sized_qty"] = size_position(pos, params)
        vega_1, dgamma_1 = _entry_greeks(pos)
        pos["sized_vega"] = abs(pos["sized_qty"]) * vega_1
        pos["sized_dgamma"] = abs(pos["sized_qty"]) * dgamma_1

    # ── 2. portfolio-level clip ──────────────────────────────────────────
    positions = portfolio_limits(positions, params)

    # ── 3. run engine per position ───────────────────────────────────────
    ledgers = []
    pos_reports = []
    for pos in positions:
        win = path[(path["date"] >= pos["date"])
                   & (path["date"] <= pos["expiry"])]
        legs = [Leg(K=pos["K"], expiry=pos["expiry"], cp=+1,
                    qty=pos["sized_qty"], mark_vol=pos["mark_vol"]),
                Leg(K=pos["K"], expiry=pos["expiry"], cp=-1,
                    qty=pos["sized_qty"], mark_vol=pos["mark_vol"])]
        led = run_hedged(win["date"], win["close"].to_numpy(), legs,
                         r=pos["r"], q=pos["q"])
        led = led[["date", "equity"]].copy()
        led.columns = ["date", "equity"]
        ledgers.append(led)
        pos_reports.append({
            "date": str(pos["date"].date()),
            "expiry": str(pos["expiry"].date()),
            "side": pos["side"],
            "K": pos["K"],
            "unit_qty": pos["qty"],
            "sized_qty": pos["sized_qty"],
            "sized_vega": pos["sized_vega"],
            "sized_dgamma": pos["sized_dgamma"],
            "mark_vol": pos["mark_vol"],
            "pnl": float(led["equity"].iloc[-1]),
        })

    # ── 4. aggregate portfolio equity ────────────────────────────────────
    all_dates = sorted(set().union(*(l["date"].tolist() for l in ledgers)))
    date_idx = pd.DatetimeIndex(all_dates)

    portfolio_eq = pd.Series(0.0, index=date_idx, name="equity")
    for led in ledgers:
        led_s = led.set_index("date")["equity"].reindex(date_idx, fill_value=0.0)
        # Before entry: 0.  After settlement: carry the final equity.
        # The engine already does this: equity starts at 0 and ends settled.
        # But for dates outside the position window, reindex fills 0 — correct
        # for pre-entry; for post-settlement we need to carry the final value.
        final_date = led["date"].iloc[-1]
        final_eq = float(led["equity"].iloc[-1])
        mask_after = date_idx > final_date
        led_s[mask_after] = final_eq
        portfolio_eq = portfolio_eq + led_s

    # ── 5. drawdown kill-switch ──────────────────────────────────────────
    portfolio_eq_killed, kill_date = drawdown_kill(
        portfolio_eq, params["max_drawdown_pct"])

    peak = portfolio_eq_killed.cummax()
    dd = peak - portfolio_eq_killed
    dd_pct = dd / peak.clip(lower=1.0)

    portfolio_df = pd.DataFrame({
        "date": date_idx,
        "equity": portfolio_eq_killed.values,
        "equity_raw": portfolio_eq.values,
        "peak": peak.values,
        "drawdown": dd.values,
        "drawdown_pct": dd_pct.values,
        "killed": date_idx >= pd.Timestamp(kill_date) if kill_date else False,
    })

    # ── 6. outputs ───────────────────────────────────────────────────────
    processed_dir.mkdir(parents=True, exist_ok=True)
    portfolio_df.to_parquet(processed_dir / "portfolio.parquet", index=False)

    total_pnl = float(portfolio_eq_killed.iloc[-1])
    max_dd = float(dd.max())
    max_dd_pct_val = float(dd_pct.max())
    summary = {
        "n_positions": len(positions),
        "sizing_params": {k: float(v) for k, v in params.items()},
        "positions": pos_reports,
        "total_pnl": total_pnl,
        "total_pnl_raw": float(portfolio_eq.iloc[-1]),
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct_val,
        "kill_date": kill_date,
        "gross_vega": float(sum(p["sized_vega"] for p in pos_reports)),
        "gross_dgamma": float(sum(p["sized_dgamma"] for p in pos_reports)),
        "date_range": [str(date_idx[0].date()), str(date_idx[-1].date())],
        "n_bars": len(portfolio_df),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="\n") as fh:
        json.dump(summary, fh, indent=2)

    # ── 7. plot ──────────────────────────────────────────────────────────
    if make_plots:
        _plot_portfolio(portfolio_df, summary, plots_dir)

    print(f"portfolio: {len(positions)} positions, "
          f"total PnL ${total_pnl:.2f}, max DD ${max_dd:.2f} "
          f"({max_dd_pct_val:.1%})"
          + (f", killed on {kill_date}" if kill_date else ""))
    print(f"-> {processed_dir / 'portfolio.parquet'}")
    print(f"-> {summary_path}")
    return summary


def _plot_portfolio(df: pd.DataFrame, summary: dict,
                    plots_dir: Path = PLOTS_DIR):
    """Equity curve + drawdown subplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                  height_ratios=[2, 1])
    ax.plot(df["date"], df["equity"], lw=1.5, label="portfolio equity (killed)"
            if summary["kill_date"] else "portfolio equity")
    if summary["kill_date"]:
        ax.plot(df["date"], df["equity_raw"], lw=0.8, ls="--", color="0.5",
                label="equity (raw, no kill)")
        ax.axvline(pd.Timestamp(summary["kill_date"]), color="red", lw=0.8,
                   ls=":", label="kill switch")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("cumulative PnL ($)")
    ax.legend(fontsize=8)
    ax.set_title(f"Day 23 — Portfolio equity ({summary['n_positions']} "
                 f"positions, PnL ${summary['total_pnl']:.1f})")

    ax2.fill_between(df["date"], 0, -df["drawdown"], color="firebrick",
                     alpha=0.4)
    ax2.set_ylabel("drawdown ($)")
    ax2.set_xlabel("date")

    p = plots_dir / "portfolio_equity.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    run_portfolio()
