"""
Phase-2 Day 36 — SPY cost sensitivity sweep (v2 robustness #2, isolated).

PLAN v2: "Cost sensitivity sweep (x0.5/1/2) -> PnL-vs-cost curve." The question
is whether the Day-34 SPY disproof (net -$8,936) is an artifact of the cost
assumption. It is not, and this shows exactly why:

  1. COST-MULTIPLIER SWEEP (x0/0.5/1/2). Net PnL is EXACTLY linear in a cost
     multiplier k: gross is cost-independent, and every option-cost component
     (half the quoted spread on both legs + $0.65/contract commission) scales
     linearly in k, so net(k) = gross - k * realized_cost is the exact x k
     cost-model result, not a linearisation. Because gross is ALREADY negative
     (-$7,493), net(k) < 0 for every k >= 0 -- the book loses even at ZERO
     transaction cost. There is no positive cost reduction that saves it.

  2. HEDGE-SLIPPAGE STRESS. The pre-registration sets underlying_slippage_bps=0
     (SPY is penny-wide). This stresses that to 1/2/5 bps. Slippage cost is
     bps/1e4 * (total hedge notional), linear in bps, so ONE engine re-run at
     1 bp recovers the notional and the rest is arithmetic. Frequent M/W/F
     hedging of a ~$450 underlying over up-to-60-day holds adds only a small
     drag, and (gross being negative) it too can only deepen the loss.

    results/phase2/costs_summary_spy.json   (base decomposition + gross)
    results/phase2/returns_summary_spy.json (capital base)
    src.backtest.costs.run_costs @ 1 bp     (hedge notional, scratch output)
      -> results/phase2/cost_sweep_spy.json

Reads the Day-34 artifacts and re-runs run_costs only to a SCRATCH path with
make_plots=False; the tracked costs_summary_spy.json and the plots are untouched.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.costs import _load_cost_params, run_costs      # noqa: E402
from src.backtest.reconcile import load_price_path               # noqa: E402
from src.utils.seed import set_global_seed                       # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "phase2" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "phase2" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
CONFIG = PROJECT_ROOT / "config" / "spy_phase2.yaml"

COST_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)
SLIPPAGE_BPS = (0.0, 1.0, 2.0, 5.0)


def _hedge_notional_usd() -> float:
    """Total |traded| * S over the whole book, via one 1-bp re-run.

    Slippage cost is linear in bps (it does not change hedge decisions), so a
    single run at 1 bp gives notional = hedge_slippage_cost * 1e4. Writes only
    to a scratch file; nothing tracked is touched.
    """
    params = _load_cost_params(CONFIG)
    params["underlying_slippage_bps"] = 1.0
    price = load_price_path(RAW_DIR, "spy_ohlc.parquet", "spy_ohlc_ext.parquet")
    with tempfile.TemporaryDirectory() as tmp:
        summ = run_costs(params=params, processed_dir=PROCESSED_DIR,
                         price_path=price, summary_path=Path(tmp) / "c.json",
                         plots_dir=Path(tmp), make_plots=False, config_path=CONFIG)
    return summ["total_hedge_slippage"] * 1e4


def main() -> None:
    set_global_seed()

    base = json.loads((RESULTS_DIR / "costs_summary_spy.json").read_text())
    cap = json.loads((RESULTS_DIR / "returns_summary_spy.json").read_text())[
        "capital_base_usd"]
    gross = base["gross_pnl"]
    realized_cost = base["total_cost"]            # half-spread + commission, bps=0

    # ── cost-multiplier sweep (exact) ─────────────────────────────────────────
    curve = []
    for k in COST_MULTIPLIERS:
        net = gross - k * realized_cost
        curve.append({
            "multiplier": k,
            "total_cost_usd": k * realized_cost,
            "net_pnl_usd": net,
            "net_return_on_capital": net / cap,
        })
    # net(k)=0  ->  k = gross/cost; gross<0 -> k<0 (no cost cut saves it)
    break_even = gross / realized_cost

    # ── hedge-slippage stress (one re-run for the notional, then arithmetic) ──
    notional = _hedge_notional_usd()
    base_net = gross - realized_cost              # slippage adds on top of base
    slip_curve = []
    for bps in SLIPPAGE_BPS:
        slip = bps / 1e4 * notional
        net = base_net - slip
        slip_curve.append({
            "underlying_slippage_bps": bps,
            "hedge_slippage_usd": slip,
            "net_pnl_usd": net,
            "net_return_on_capital": net / cap,
        })

    out = {
        "base": {
            "gross_pnl_usd": gross,
            "half_spread_usd": base["total_half_spread"],
            "commission_usd": base["total_commission"],
            "realized_cost_usd": realized_cost,
            "net_pnl_usd": base["net_pnl"],
            "capital_base_usd": cap,
            "cost_as_pct_of_premium": base["cost_as_pct_of_premium"],
        },
        "cost_multiplier_sweep": {
            "note": "net(k) = gross - k * realized_cost. EXACT, not a "
                    "linearisation: gross is cost-independent and every "
                    "option-cost component scales linearly in the multiplier, "
                    "so this is the x0.5/1/2 cost-model result.",
            "curve": curve,
            "break_even_multiplier": break_even,
            "finding": "gross is already negative (-$7,493), so net < 0 at every "
                       "k >= 0 -- the book loses even at ZERO transaction cost "
                       "(break-even multiplier is negative). The Day-34 SPY "
                       "disproof is NOT a transaction-cost artifact; costs only "
                       "deepen a loss that exists gross.",
        },
        "hedge_slippage_stress": {
            "note": "pre-registration sets underlying_slippage_bps=0 (SPY "
                    "penny-wide); this stresses it. slippage = bps/1e4 * total "
                    "hedge notional, linear in bps (one 1-bp engine re-run gives "
                    "the notional).",
            "total_hedge_notional_usd": notional,
            "hedge_notional_over_capital": notional / cap,
            "curve": slip_curve,
            "finding": "the M/W/F hedge programme turns over ${:,.0f} of "
                       "underlying notional -- {:.0f}x the ${:,.0f} capital base "
                       "-- so unlike the option costs the book IS sensitive to "
                       "underlying slippage. At SPY's realistic penny-wide "
                       "half-spread (~0.1 bp on a ~$450 stock) the drag is only "
                       "~$500; the 1-5 bps stress is deliberately harsh and "
                       "materially deepens the loss (to -5% .. -13%), but never "
                       "flips the sign -- the book is negative before any "
                       "slippage. The cost-sensitive dimension here is hedge "
                       "TURNOVER, not the option spread or commission.".format(
                           notional, notional / cap, cap),
        },
    }
    out_path = RESULTS_DIR / "cost_sweep_spy.json"
    out_path.write_text(json.dumps(out, indent=2), newline="\n")

    print(f"\nbase: gross ${gross:.2f} | cost ${realized_cost:.2f} "
          f"(half-spread ${base['total_half_spread']:.0f} + commission "
          f"${base['total_commission']:.0f}) | net ${base['net_pnl']:.2f}")
    print("\ncost-multiplier sweep (exact x k cost model):")
    for c in curve:
        print(f"  x{c['multiplier']:<4}: cost ${c['total_cost_usd']:8.2f} | "
              f"net ${c['net_pnl_usd']:+9.2f} | "
              f"{c['net_return_on_capital']:+.2%}")
    print(f"  break-even multiplier k* = {break_even:+.2f} "
          f"(<0 -> no cost cut saves it; loses at zero cost)")
    print(f"\nhedge-slippage stress (notional ${notional:,.0f}):")
    for c in slip_curve:
        print(f"  {c['underlying_slippage_bps']:>4.1f} bps: slippage "
              f"${c['hedge_slippage_usd']:7.2f} | net "
              f"${c['net_pnl_usd']:+9.2f} | {c['net_return_on_capital']:+.2%}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
