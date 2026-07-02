"""Run the full independent-verification suite (Days 3-14 + project audit).

These scripts re-derive results with inline/independent implementations
(erf pricers, finite differences, complex-step, Breeden-Litzenberger
densities) rather than the project's own code paths, so they catch
compensating bugs the pytest suite can miss. Run after any pipeline change:

    python scripts/verify/run_all.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "fd_check_day3.py",      # Greeks vs finite differences
    "cs_check_day4.py",      # vanna/volga vs complex-step
    "verify_day5.py",        # IV inversion vs independent erf pricer
    "verify_day7.py",        # forwards vs actual AAPL closes
    "verify_day10.py",       # SVI RMSE recompute + quadratic baseline
    "verify_day11.py",       # butterfly via BL density
    "verify_day12.py",       # constrained refit + Vogt hard case
    "verify_day13.py",       # calendar off-node grid + call-price monotonicity
    "verify_day14.py",       # assembly vs inline reimplementation
    "audit_full.py",         # cleaning invariants, spine, results coherence
]

HERE = Path(__file__).resolve().parent


def main() -> int:
    failed = []
    for name in SCRIPTS:
        r = subprocess.run([sys.executable, str(HERE / name)],
                           capture_output=True, text=True,
                           cwd=HERE.parent.parent)
        status = "PASS" if r.returncode == 0 else "FAIL"
        print(f"[{status}] {name}")
        if r.returncode != 0:
            failed.append(name)
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
    print(f"\n{len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} verification scripts passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
