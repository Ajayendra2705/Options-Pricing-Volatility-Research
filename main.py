"""
Options Vol-Arb Main Entry Point
================================
Run this to reproduce all results from raw data.
"""

import argparse
import sys
from pathlib import Path

from src.utils.seed import set_global_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Options Vol-Arb Pipeline")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "clean", "surface", "backtest", "report"],
                        help="Pipeline stage to run")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed()
    
    print(f"Starting pipeline... Stage: {args.stage}")

    if args.stage in ("all", "clean"):
        from src.surface.clean import run_cleaning
        run_cleaning()

    if args.stage in ("all", "surface"):
        # Part 1 pipeline (Days 7-14): forwards -> IVs -> SVI diagnostics ->
        # joint arb-free fit -> assembled surface + QC
        from src.surface.forwards import run_forwards
        from src.surface.iv_surface import run_iv_surface
        from src.surface.svi import run_svi_all
        from src.surface.no_arb import run_arb_check, run_constrained_refit
        from src.surface.assemble import run_assembly
        run_forwards()
        run_iv_surface()
        run_svi_all()
        run_constrained_refit()   # Day-12 per-slice diagnostic + butterfly log
        run_arb_check()           # Day-13 joint fit (authoritative surface)
        run_assembly()

    if args.stage in ("all", "backtest"):
        # Part 2 pipeline (Days 16+): realized vol -> HAR forecast -> signal
        # -> positions + attribution reconciliation (Day-22 gate)
        from src.backtest.realized_vol import run_realized_vol
        from src.backtest.har import run_har
        from src.backtest.signal import run_signal
        from src.backtest.reconcile import run_reconcile
        from src.backtest.portfolio import run_portfolio
        from src.backtest.costs import run_costs
        from src.backtest.returns import run_returns
        from src.backtest.metrics import run_metrics
        from src.backtest.alpha import run_alpha
        from src.backtest.stats import run_stats
        run_realized_vol()
        run_har()
        run_signal()
        run_reconcile()
        run_portfolio()
        run_costs()
        run_returns()
        run_metrics()
        run_alpha()
        run_stats()

    # TODO: Implement stages
    # if args.stage in ("all", "report"):
    #     generate_report()

    print("Done.")

if __name__ == "__main__":
    main()
