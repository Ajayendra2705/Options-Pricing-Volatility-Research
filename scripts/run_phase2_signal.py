"""
Phase-2 Day 33 — RV / HAR / signal on SPY, isolated from v1.

Same code as v1's backtest-front (Days 16-18), pointed at Phase-2 paths via the
Day-32 seams:

    data/phase2/raw/spy_ohlc.parquet + data/phase2/processed/svi_params_joint.parquet
      -> data/phase2/processed/{realized_vol,har_forecast,signal}.parquet
      -> results/phase2/{har_stats_spy,signal_summary_spy}.json
      -> results/phase2/plots/*.png   (3 figures, untracked)

Nothing here may touch v1: every default in src/backtest/** is v1's path, and
this driver only ever passes Phase-2 paths. Config: config/spy_phase2.yaml
(pre-registered; signal/side rules identical to v1's primary.yaml).

Prerequisite: scripts/run_phase2_surface.py (Day 32) has produced the joint
arb-free SVI fits the ATM-IV leg reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.har import run_har                     # noqa: E402
from src.backtest.realized_vol import run_realized_vol   # noqa: E402
from src.backtest.signal import run_signal               # noqa: E402
from src.utils.seed import set_global_seed               # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "phase2" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "phase2" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
PLOTS_DIR = RESULTS_DIR / "plots"

TICKER = "SPY"
OHLC = RAW_DIR / "spy_ohlc.parquet"
HAR_STATS = RESULTS_DIR / "har_stats_spy.json"
SIGNAL_SUMMARY = RESULTS_DIR / "signal_summary_spy.json"
CONFIG_LABEL = "config/spy_phase2.yaml (pre-registered)"


def main() -> None:
    set_global_seed()
    joint = PROCESSED_DIR / "svi_params_joint.parquet"
    if not joint.exists():
        raise FileNotFoundError(
            f"{joint} missing — run scripts/run_phase2_surface.py (Day 32) first")

    run_realized_vol(ohlc_path=OHLC, processed_dir=PROCESSED_DIR,
                     plots_dir=PLOTS_DIR, ticker=TICKER)
    run_har(ohlc_path=OHLC, processed_dir=PROCESSED_DIR,
            plots_dir=PLOTS_DIR, stats_path=HAR_STATS)
    tab = run_signal(ohlc_path=OHLC, processed_dir=PROCESSED_DIR,
                     plots_dir=PLOTS_DIR, summary_path=SIGNAL_SUMMARY,
                     config_label=CONFIG_LABEL)

    n_dates = tab["date"].nunique()
    n_sig = int(tab["signal_raw"].notna().sum())
    print(f"\nphase2 signal ({TICKER}): {n_dates} dates, {len(tab)} slices, "
          f"{n_sig} with signal | sides "
          f"{tab['side'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
