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

    # TODO: Implement stages
    # if args.stage in ("all", "surface"):
    #     run_surface()
    # if args.stage in ("all", "backtest"):
    #     run_backtest()
    # if args.stage in ("all", "report"):
    #     generate_report()
        
    print("Done.")

if __name__ == "__main__":
    main()
