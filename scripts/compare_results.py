"""
Cross-platform reproducibility check.

`git diff --exit-code` is the wrong gate for this project's artifacts, and CI on
ubuntu proved it: rerunning the pipeline on a different OS reproduces the same
NUMBERS but not the same BYTES, for two reasons that are not defects —

  * floats: SLSQP/least-squares land a few ulps apart under a different
    BLAS/LAPACK, and the difference propagates through the surface into every
    downstream figure (observed: ~1e-9 relative);
  * plots: matplotlib PNG bytes depend on the platform's font rasterization.

So byte-equality holds within a platform (that check still runs, locally and in
CI, as "the pipeline is deterministic"), while ACROSS platforms the honest claim
is numeric equality to a stated tolerance. This script is that claim, enforced:
it compares every tracked results JSON on disk against the committed version and
fails if any number moves by more than `--rtol`, or if any string/structure
changes at all.

Usage:
    python scripts/compare_results.py                 # vs HEAD, rtol 1e-6
    python scripts/compare_results.py --rtol 1e-9
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def committed_json(rel_path: str, ref: str) -> dict:
    """The version of `rel_path` recorded at `ref` (e.g. HEAD)."""
    out = subprocess.run(
        ["git", "show", f"{ref}:{rel_path}"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def compare(old, new, rtol: float, atol: float, path: str = "") -> list[str]:
    """Recursively diff two JSON trees. Numbers within tolerance are equal;
    everything else must match exactly. Returns a list of human-readable diffs."""
    if isinstance(old, dict) and isinstance(new, dict):
        diffs = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                diffs.append(f"{path}/{key}: added")
            elif key not in new:
                diffs.append(f"{path}/{key}: REMOVED")
            else:
                diffs += compare(old[key], new[key], rtol, atol, f"{path}/{key}")
        return diffs

    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} -> {len(new)}"]
        diffs = []
        for i, (o, n) in enumerate(zip(old, new)):
            diffs += compare(o, n, rtol, atol, f"{path}[{i}]")
        return diffs

    # bool is an int subclass — check it first, and never tolerance-compare it
    if isinstance(old, bool) or isinstance(new, bool):
        return [] if old == new else [f"{path}: {old} -> {new}"]

    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if math.isnan(old) and math.isnan(new):
            return []
        if math.isclose(old, new, rel_tol=rtol, abs_tol=atol):
            return []
        denom = max(abs(old), abs(new), atol)
        return [f"{path}: {old!r} -> {new!r} (rel {abs(old - new) / denom:.2e})"]

    return [] if old == new else [f"{path}: {old!r} -> {new!r}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="HEAD", help="git ref to compare against")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance for numbers (default 1e-6)")
    ap.add_argument("--atol", type=float, default=1e-12)
    args = ap.parse_args()

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("no results JSONs found — run `python main.py` first")
        return 1

    failed = {}
    worst = 0.0
    for f in files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        try:
            old = committed_json(rel, args.ref)
        except subprocess.CalledProcessError:
            print(f"  [skip]  {rel} (not tracked at {args.ref})")
            continue
        new = json.loads(f.read_text())
        diffs = compare(old, new, args.rtol, args.atol)
        if diffs:
            failed[rel] = diffs
            print(f"  [DIFF]  {rel} — {len(diffs)} field(s)")
        else:
            print(f"  [ok]    {rel}")

    if failed:
        print(f"\nFAIL: regenerated results differ from {args.ref} "
              f"beyond rtol={args.rtol:g}\n")
        for rel, diffs in failed.items():
            print(f"{rel}:")
            for d in diffs[:10]:
                print(f"    {d}")
            if len(diffs) > 10:
                print(f"    ... and {len(diffs) - 10} more")
        return 1

    print(f"\nOK: every tracked results JSON matches {args.ref} "
          f"within rtol={args.rtol:g} ({len(files)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
