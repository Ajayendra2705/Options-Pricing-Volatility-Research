"""
Cross-platform reproducibility check.

`git diff --exit-code` is the wrong gate for these artifacts, and CI on ubuntu
proved it. Rerunning the pipeline on another OS gives the same conclusions but
not the same bytes:

  * the constrained SVI optimizers (SLSQP / least-squares) minimize a flat
    objective, so under a different BLAS they settle on a slightly different
    point of the optimum: fitted marks move ~1e-5 relative (~0.001 vol pts);
  * matplotlib rasterizes PNGs differently.

Downstream, that ~1e-5 on a mark is AMPLIFIED wherever the result is a small
difference of large numbers. `gross_pnl` is the extreme case: +$505 of short-vol
legs against -$502 of long-vol legs nets to ~$3.8, so a one-cent move in a leg is
a ~2% move in the headline. Reproducibility here is therefore a claim about
DOLLARS (cents), not about relative error — a uniform rtol would either fail on
gross_pnl or be so loose it gates nothing.

So this script enforces two things instead:

  1. `--rtol` / `--atol` on every field: a number passes if it is within EITHER
     the relative tolerance OR the absolute one. Defaults (1e-3, $0.10) are set
     so BLAS-scale drift passes and any real code regression — which moves things
     by far more — fails. Strings, structure and booleans must match exactly.

  2. `--conclusions`: the claims the project actually makes must still hold on
     the fresh platform (net PnL negative, surface arb-free, attribution gate,
     alpha statistically zero, Sharpe CI spanning zero). A result that survives a
     tolerance check but flips a conclusion is not reproduced, and this catches
     that; a result whose 8th decimal moved is reproduced, and this ignores it.

Usage:
    python scripts/compare_results.py                        # vs HEAD
    python scripts/compare_results.py --conclusions          # + claims still hold
    python scripts/compare_results.py --rtol 1e-9 --atol 0   # strict (same platform)
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

    # bool is an int subclass, so `True == 1.0` — check it first and compare
    # exactly, or a flipped arb-free flag could slip through as float noise
    if isinstance(old, bool) or isinstance(new, bool):
        same = isinstance(old, bool) and isinstance(new, bool) and old == new
        return [] if same else [f"{path}: {old!r} -> {new!r}"]

    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if math.isnan(old) and math.isnan(new):
            return []
        if math.isclose(old, new, rel_tol=rtol, abs_tol=atol):
            return []
        denom = max(abs(old), abs(new), atol)
        return [f"{path}: {old!r} -> {new!r} (rel {abs(old - new) / denom:.2e})"]

    return [] if old == new else [f"{path}: {old!r} -> {new!r}"]


def check_conclusions(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Every claim the project makes, re-asserted against the regenerated results.

    These are the project's conclusions, not its decimals: if a platform, a
    library bump or a refactor breaks one of these, the result did not reproduce
    in any sense that matters.
    """
    j = lambda name: json.loads((results_dir / f"{name}.json").read_text())
    arb, qc = j("arb_violations"), j("surface_qc")
    rec, costs, m = j("attribution_reconcile"), j("costs_summary"), j("metrics")
    sh = m["statistical_honesty"]
    reg = m["alpha_regression"]
    ci = sh["bootstrap_ci_95"]

    claims = [
        # surface is arbitrage-free (Days 11-14)
        ("surface: no butterfly violations",
         arb["butterfly"]["n_violations"] == 0),
        ("surface: no calendar violations",
         arb["calendar"]["n_pairs_violated"] == 0),
        ("surface: interpolation arb-free",
         qc["all_interp_butterfly_ok"] and qc["all_interp_calendar_ok"]),
        ("surface: >95% of quotes within 1 vol pt",
         qc["frac_within_1volpt"] > 0.95),
        # attribution reconciles to the ledger (Day 22 gate)
        ("attribution: book residual < 20% of sum|daily PnL|",
         rec["book_residual_abs_share"] < 0.20),
        ("attribution: worst position residual < 10% of premium",
         rec["worst_position_residual_over_premium"] < 0.10),
        # the disproof itself (Days 24-28)
        ("costs: net PnL is negative", costs["net_pnl"] < 0),
        ("costs: costs exceed gross PnL",
         costs["total_cost"] > costs["gross_pnl"]),
        ("returns: net return on capital is negative",
         m["net_return_on_capital"] < 0),
        ("stats: Sharpe is not significant (|NW t| < 2)",
         abs(sh["sharpe"]["nw_tstat"]) < 2),
        ("stats: bootstrap 95% CI for Sharpe spans zero",
         ci["ci_2.5"] < 0 < ci["ci_97.5"]),
        ("alpha: no edge vs the VRP factor (|NW t| < 2)",
         abs(reg["alpha_t"]) < 2),
        ("alpha: book loads negatively on a short-vol factor (net long vol)",
         reg["beta"] < 0),
    ]
    return [f"CLAIM BROKEN: {label}" for label, ok in claims if not ok]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="HEAD", help="git ref to compare against")
    ap.add_argument("--rtol", type=float, default=1e-3,
                    help="relative tolerance for numbers (default 1e-3)")
    ap.add_argument("--atol", type=float, default=0.10,
                    help="absolute tolerance, i.e. dollars (default 0.10)")
    ap.add_argument("--conclusions", action="store_true",
                    help="also assert the project's claims still hold")
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
        print(f"\nFAIL: regenerated results differ from {args.ref} beyond "
              f"rtol={args.rtol:g} / atol={args.atol:g}\n")
        for rel, diffs in failed.items():
            print(f"{rel}:")
            for d in diffs[:10]:
                print(f"    {d}")
            if len(diffs) > 10:
                print(f"    ... and {len(diffs) - 10} more")
        return 1

    print(f"\nOK: every tracked results JSON matches {args.ref} within "
          f"rtol={args.rtol:g} / atol={args.atol:g} ({len(files)} files).")

    if args.conclusions:
        broken = check_conclusions()
        if broken:
            print("\nFAIL: the results reproduced numerically but a CLAIM changed:\n")
            for b in broken:
                print(f"    {b}")
            return 1
        print("OK: every claim the project makes still holds on this platform.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
