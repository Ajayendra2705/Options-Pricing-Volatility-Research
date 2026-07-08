"""
Day 24 — Transaction costs: gross vs net PnL.

Applies the PRE-REGISTERED cost model (config/primary.yaml `costs:` block,
locked Day 18) to the pre-registered unit-qty straddle book (the same 10
positions Day 22 built and gated) and reports gross vs net PnL.

COST COMPONENTS (per straddle = call leg + put leg):
  1. Option entry half-spread — cross half the quoted effective spread on
     BOTH legs at entry: 0.5*(ask-bid) per share * mult * |qty|, summed over
     call+put. Crossing costs money whether opening long or short, so it is
     always a positive drag.  Marked from the entry-date chain_clean quotes.
  2. Commission — $0.65 per contract, both legs, on the opening trade:
     2 * 0.65 * |qty|.  Held to expiry -> cash-settled intrinsic, no closing
     trade, so no closing commission (documented assumption).
  3. Hedge slippage on the underlying — config declares AAPL penny-wide,
     ZERO spread in the primary (`underlying_fill: close, zero spread`).  The
     mechanism is implemented and parametrized (`underlying_slippage_bps`,
     default 0) so a robustness sweep can stress it without touching primary.

WHY THE UNIT-QTY BOOK: this is the book the signal defines and Day 22 gated
(gross +$3.87, shorts and longs nearly cancel).  Costs scale with |qty|, so
applying them here gives the honest, un-rescaled gross-vs-net comparison that
is the Day-24 deliverable.  The Day-23 risk-sized/kill-switched portfolio is a
separate object (returns land Day 25).

NO LOOKAHEAD: entry costs use only the entry-date quotes; hedge slippage uses
only the ledger's realized `traded` shares.

OUTPUT:
  - results/costs_summary.json  (tracked, byte-stable)
  - results/plots/gross_vs_net.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest.engine import Leg, run_hedged
from src.backtest.reconcile import build_positions, load_price_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFIG_PATH = PROJECT_ROOT / "config" / "primary.yaml"

MULT = 100.0


# ── config ─────────────────────────────────────────────────────────────────

def _load_cost_params() -> dict:
    """Parse the `costs:` block from primary.yaml in isolation.

    The full file mixes prose with unquoted colons (unparseable by strict
    YAML), so — like `portfolio._load_sizing_params` — we slice out just the
    costs block.  `underlying_slippage_bps` is not in the pre-registered file
    (primary declares zero underlying spread); it defaults to 0 here and is a
    robustness-only knob.
    """
    defaults = {
        "commission_per_contract_usd": 0.65,
        "underlying_slippage_bps": 0.0,   # primary: AAPL penny-wide, zero
    }
    if CONFIG_PATH.exists():
        try:
            lines = CONFIG_PATH.read_text().splitlines()
            block, in_block = [], False
            for line in lines:
                if line.strip().startswith("costs:"):
                    in_block = True
                    block.append(line)
                    continue
                if in_block:
                    if line and not line[0].isspace() and not line.startswith("#"):
                        break
                    block.append(line)
            if block:
                cfg = yaml.safe_load("\n".join(block))
                if cfg and "costs" in cfg:
                    c = cfg["costs"]
                    if "commission_per_contract_usd" in c:
                        defaults["commission_per_contract_usd"] = float(
                            c["commission_per_contract_usd"])
        except Exception:
            pass
    return defaults


# ── cost components ─────────────────────────────────────────────────────────

def option_entry_cost(pos: dict, chain: pd.DataFrame, params: dict) -> dict:
    """Half-spread + commission for one straddle at entry.

    Returns {half_spread, commission, total} in dollars (>= 0).
    """
    sl = chain[(chain["date"] == pos["date"])
               & (chain["expiry"] == pos["expiry"])
               & (chain["strike"] == pos["K"])]
    call = sl[sl["option_type"] == "C"]
    put = sl[sl["option_type"] == "P"]
    if call.empty or put.empty:
        raise RuntimeError(f"missing entry quote for straddle {pos['date']}")
    c, p = call.iloc[0], put.iloc[0]

    qty = abs(pos["qty"])
    half_c = 0.5 * (c["ask"] - c["bid"])          # per share
    half_p = 0.5 * (p["ask"] - p["bid"])
    half_spread = (half_c + half_p) * MULT * qty  # both legs

    commission = 2.0 * params["commission_per_contract_usd"] * qty
    return {
        "half_spread": float(half_spread),
        "commission": float(commission),
        "total": float(half_spread + commission),
    }


def hedge_slippage_cost(ledger: pd.DataFrame, params: dict) -> float:
    """Slippage on the underlying rebalancing trades.

    cost = (bps/1e4) * sum |traded_t| * S_t.  Primary bps=0 -> 0.
    """
    bps = params.get("underlying_slippage_bps", 0.0)
    if bps == 0.0:
        return 0.0
    notional = float((ledger["traded"].abs() * ledger["S"]).sum())
    return bps / 1e4 * notional


# ── runner ──────────────────────────────────────────────────────────────────

def run_costs(params: dict | None = None) -> dict:
    """Gross vs net PnL on the pre-registered unit-qty book.

    Returns the summary dict (also written to costs_summary.json).
    """
    if params is None:
        params = _load_cost_params()

    positions = build_positions()
    path = load_price_path()
    chain = pd.read_parquet(PROCESSED_DIR / "chain_clean.parquet")

    reports = []
    for pos in positions:
        win = path[(path["date"] >= pos["date"])
                   & (path["date"] <= pos["expiry"])]
        legs = [Leg(K=pos["K"], expiry=pos["expiry"], cp=+1, qty=pos["qty"],
                    mark_vol=pos["mark_vol"]),
                Leg(K=pos["K"], expiry=pos["expiry"], cp=-1, qty=pos["qty"],
                    mark_vol=pos["mark_vol"])]
        led = run_hedged(win["date"], win["close"].to_numpy(), legs,
                         r=pos["r"], q=pos["q"])
        gross = float(led["equity"].iloc[-1])

        ec = option_entry_cost(pos, chain, params)
        hs = hedge_slippage_cost(led, params)
        total_cost = ec["total"] + hs
        net = gross - total_cost

        reports.append({
            "date": str(pos["date"].date()),
            "expiry": str(pos["expiry"].date()),
            "side": pos["side"], "K": pos["K"],
            "premium": float(abs(led["V_opt"].iloc[0])),
            "gross_pnl": gross,
            "half_spread_cost": ec["half_spread"],
            "commission_cost": ec["commission"],
            "hedge_slippage_cost": float(hs),
            "total_cost": float(total_cost),
            "net_pnl": float(net),
        })

    gross_total = float(sum(r["gross_pnl"] for r in reports))
    cost_total = float(sum(r["total_cost"] for r in reports))
    net_total = float(sum(r["net_pnl"] for r in reports))
    summary = {
        "n_positions": len(reports),
        "cost_params": params,
        "positions": reports,
        "gross_pnl": gross_total,
        "total_cost": cost_total,
        "total_half_spread": float(sum(r["half_spread_cost"] for r in reports)),
        "total_commission": float(sum(r["commission_cost"] for r in reports)),
        "total_hedge_slippage": float(sum(r["hedge_slippage_cost"]
                                          for r in reports)),
        "net_pnl": net_total,
        "cost_as_pct_of_premium": float(
            cost_total / sum(r["premium"] for r in reports)),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "costs_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    _plot_gross_vs_net(reports, summary)

    print(f"costs: {len(reports)} positions | gross ${gross_total:.2f} "
          f"- costs ${cost_total:.2f} = net ${net_total:.2f}")
    print(f"  half-spread ${summary['total_half_spread']:.2f}, "
          f"commission ${summary['total_commission']:.2f}, "
          f"hedge slippage ${summary['total_hedge_slippage']:.2f}")
    print(f"-> {RESULTS_DIR / 'costs_summary.json'}")
    return summary


def _plot_gross_vs_net(reports: list[dict], summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['date']}\n{r['expiry']}" for r in reports]
    x = np.arange(len(reports))
    w = 0.4
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                  gridspec_kw={"width_ratios": [3, 1]})
    ax.bar(x - w / 2, [r["gross_pnl"] for r in reports], w,
           label="gross", color="steelblue")
    ax.bar(x + w / 2, [r["net_pnl"] for r in reports], w,
           label="net (after costs)", color="firebrick")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, labels, fontsize=6, rotation=45)
    ax.set_ylabel("PnL ($)")
    ax.set_title("Per-position gross vs net PnL")
    ax.legend(fontsize=8)

    ax2.bar([0, 1], [summary["gross_pnl"], summary["net_pnl"]],
            color=["steelblue", "firebrick"])
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_xticks([0, 1], ["gross", "net"])
    ax2.set_title(f"Book: ${summary['gross_pnl']:.0f} -> "
                  f"${summary['net_pnl']:.0f}")
    p = PLOTS_DIR / "gross_vs_net.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    run_costs()
