"""
Phase-2 Day 32 — run v1's surface stage on SPY, isolated from v1.

Same code as `python main.py --stage surface`, pointed at Phase-2 paths via the
output-dir seams on the runners:

    data/phase2/raw/spy_options.parquet
      -> data/phase2/processed/{chain_clean,forwards,iv_surface,svi_params*}.parquet
      -> results/phase2/{surface_qc_spy,arb_violations_spy,svi_butterfly_log_spy}.json
                          + data_quality_spy.json  (Day-31 audit block + cleaning block)

Nothing here may touch v1: every default in src/surface/** is v1's path, and this
driver only ever passes Phase-2 paths. The OHLC bars sitting in the same raw dir
are excluded by CHAIN_GLOB (the Day-30 footgun) — but they ARE passed as the
trading calendar: the DB quotes SPY on market holidays, and a date with no bar
is not a session (Day 32; 8 of the 155 raw dates).

Plots are off by default: 155 quote dates would emit ~465 figures for a stage
whose deliverable is the QC json. --plots turns them on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.surface.assemble import run_assembly            # noqa: E402
from src.surface.clean import run_cleaning               # noqa: E402
from src.surface.forwards import run_forwards            # noqa: E402
from src.surface.iv_surface import run_iv_surface        # noqa: E402
from src.surface.no_arb import run_arb_check, run_constrained_refit  # noqa: E402
from src.surface.svi import run_svi_all                  # noqa: E402
from src.utils.seed import set_global_seed               # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "phase2" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "phase2" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
PLOTS_DIR = RESULTS_DIR / "plots"

TICKER = "SPY"
QC_PATH = RESULTS_DIR / "surface_qc_spy.json"
DQ_PATH = RESULTS_DIR / "data_quality_spy.json"
ARB_PATH = RESULTS_DIR / "arb_violations_spy.json"
BFLY_PATH = RESULTS_DIR / "svi_butterfly_log_spy.json"


def parse_args():
    p = argparse.ArgumentParser(description="Phase-2 SPY surface stage")
    p.add_argument("--plots", action="store_true",
                   help="also write the per-date figures (~465 for the full window)")
    p.add_argument("--skip-clean", action="store_true",
                   help="reuse an existing data/phase2/processed/chain_clean.parquet")
    return p.parse_args()


def main() -> dict:
    args = parse_args()
    set_global_seed()
    plots = args.plots

    if not args.skip_clean:
        run_cleaning(raw_dir=RAW_DIR,
                     out_path=PROCESSED_DIR / "chain_clean.parquet",
                     results_dir=RESULTS_DIR,
                     dq_path=DQ_PATH,
                     sessions_path=RAW_DIR / "spy_ohlc.parquet")

    run_forwards(processed_dir=PROCESSED_DIR, plots_dir=PLOTS_DIR, make_plots=plots)
    run_iv_surface(processed_dir=PROCESSED_DIR, plots_dir=PLOTS_DIR, make_plots=plots)
    run_svi_all(processed_dir=PROCESSED_DIR, plots_dir=PLOTS_DIR, make_plots=plots)
    run_constrained_refit(processed_dir=PROCESSED_DIR, log_path=BFLY_PATH)
    report = run_arb_check(processed_dir=PROCESSED_DIR, report_path=ARB_PATH)
    qc = run_assembly(processed_dir=PROCESSED_DIR, plots_dir=PLOTS_DIR,
                      qc_path=QC_PATH, make_plots=plots, ticker=TICKER)

    print(f"\nphase2 surface ({TICKER}): {qc['n_dates']} dates, "
          f"{qc['n_slices_total']} slices | "
          f"butterfly violations {report['butterfly']['n_violations']} | "
          f"calendar pairs violated {report['calendar']['n_pairs_violated']}"
          f"/{report['calendar']['n_pairs_checked']} | "
          f"median RMSE {qc['rmse_iv_median'] * 100:.2f} volpts | "
          f"within 1 volpt {qc['frac_within_1volpt']:.1%}")
    return qc


if __name__ == "__main__":
    main()
