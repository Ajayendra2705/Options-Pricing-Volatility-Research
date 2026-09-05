"""
Cross-platform reproducibility check.

`git diff --exit-code` is the wrong gate for these artifacts, and CI on ubuntu
proved it. Rerunning the pipeline on another OS gives the same conclusions but
not the same bytes:

  * the constrained SVI / least-squares fits minimize a near-flat objective, so
    under a different BLAS the optimizer settles on a different point of it.
    Most slices move ~1e-5 relative (~0.001 vol pts); at least one AAPL slice
    (2023-06-09, K=180) is genuinely bistable and lands ~1 vol pt away, which
    moves that leg's premium ~2%;
  * matplotlib rasterizes PNGs differently.

Downstream, that is AMPLIFIED wherever a result is a small difference of large
numbers. `gross_pnl` is the extreme case: ~+$500 of short-vol legs against
~-$500 of long-vol legs nets near zero, so a ~$17 move in one leg is the whole
headline. `gross_pnl` therefore has NO stable significant digit across BLAS
implementations, and no rtol/atol can gate it without either failing on drift or
being so loose it gates nothing.

So across platforms this script gates the two things that ARE invariant, and
reports the rest:

  1. `--cross-platform`: structure, strings and booleans must match `--ref`
     exactly (a renamed field, a flipped arb-free flag, a Windows path baked
     into an artifact, a changed list length — all fail). Numbers are expected
     to drift and are printed, not failed.

  2. `--conclusions` (implied by `--cross-platform`): every claim the project
     makes must still hold on the fresh platform — net PnL negative, book
     near-flat gross, surface arb-free, attribution gate passed, alpha
     statistically zero, Sharpe CI spanning zero. A result that drifts within
     tolerance but flips a conclusion is not reproduced, and this catches it.

Without `--cross-platform` the check is strict: `--rtol` / `--atol` on every
field (a number passes if within EITHER), for use on the platform the artifacts
were built on, where "did you forget to rerun main.py" is a real question.

Usage:
    python scripts/compare_results.py                        # strict, vs HEAD
    python scripts/compare_results.py --conclusions          # + claims still hold
    python scripts/compare_results.py --cross-platform       # other OS/BLAS: structure + claims
    python scripts/compare_results.py --rtol 1e-9 --atol 0   # byte-strict (same platform)
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


def structural_diffs(old, new, path: str = "") -> list[str]:
    """The part of a JSON tree that must be identical on ANY platform: keys,
    list lengths, strings, booleans, and the type of every leaf. Numbers are
    deliberately ignored — a different BLAS moves them and that is drift, not a
    regression (see the module docstring). Same output format as `compare`, so
    a structural change reported here also appears in `compare`'s output.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                out.append(f"{path}/{key}: added")
            elif key not in new:
                out.append(f"{path}/{key}: REMOVED")
            else:
                out += structural_diffs(old[key], new[key], f"{path}/{key}")
        return out

    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} -> {len(new)}"]
        out = []
        for i, (o, n) in enumerate(zip(old, new)):
            out += structural_diffs(o, n, f"{path}[{i}]")
        return out

    # bool before int (bool is an int subclass): a flipped flag must never pass
    if isinstance(old, bool) or isinstance(new, bool):
        same = isinstance(old, bool) and isinstance(new, bool) and old == new
        return [] if same else [f"{path}: {old!r} -> {new!r}"]

    # two numbers: any move is drift, not structural — ignore it here
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return []

    # strings, and type changes (number -> string, etc.) must match exactly
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
        # the book is near-flat gross — the VRP is visible but not exploitable.
        # gross PnL is a ~$500-vs-$500 cancellation, so this is gated as a
        # fraction of premium traded (BLAS-stable), not to the dollar.
        ("costs: book is near-flat gross (|gross PnL| < 2% of premium traded)",
         abs(costs["gross_pnl"])
         < 0.02 * sum(abs(p["premium"]) for p in costs["positions"])),
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
    ap.add_argument("--cross-platform", action="store_true",
                    help="artifacts were built on another OS/BLAS: gate on structure "
                         "+ claims only, report numeric drift without failing on it")
    args = ap.parse_args()

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print("no results JSONs found — run `python main.py` first")
        return 1

    numeric_fail, structural_fail, drift = {}, {}, {}
    for f in files:
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        try:
            old = committed_json(rel, args.ref)
        except subprocess.CalledProcessError:
            print(f"  [skip]  {rel} (not tracked at {args.ref})")
            continue
        new = json.loads(f.read_text())

        sdiffs = structural_diffs(old, new)
        num_only = [d for d in compare(old, new, args.rtol, args.atol) if d not in sdiffs]

        if sdiffs:
            structural_fail[rel] = sdiffs
            print(f"  [CHANGED] {rel} — {len(sdiffs)} structural change(s)")
        elif num_only and args.cross_platform:
            drift[rel] = num_only
            print(f"  [drift]   {rel} — {len(num_only)} number(s) moved (different BLAS)")
        elif num_only:
            numeric_fail[rel] = num_only
            print(f"  [DIFF]  {rel} — {len(num_only)} field(s)")
        else:
            print(f"  [ok]    {rel}")

    ok = True

    if structural_fail:
        ok = False
        print(f"\nFAIL: structure / strings / flags differ from {args.ref} — "
              f"a real change, not float drift.\n")
        for rel, diffs in structural_fail.items():
            print(f"{rel}:")
            for d in diffs[:20]:
                print(f"    {d}")

    if numeric_fail:
        ok = False
        print(f"\nFAIL: regenerated results differ from {args.ref} beyond "
              f"rtol={args.rtol:g} / atol={args.atol:g}\n")
        for rel, diffs in numeric_fail.items():
            print(f"{rel}:")
            for d in diffs[:10]:
                print(f"    {d}")
            if len(diffs) > 10:
                print(f"    ... and {len(diffs) - 10} more")

    if drift:
        n = sum(len(v) for v in drift.values())
        print(f"\n{n} number(s) drift beyond rtol={args.rtol:g}/atol={args.atol:g} on this "
              f"platform — expected: the constrained-fit landing point is BLAS-dependent "
              f"(see module docstring). Not a failure; the claims are the gate.")
        for rel, diffs in drift.items():
            print(f"  {rel}:")
            for d in diffs[:8]:
                print(f"    {d}")
            if len(diffs) > 8:
                print(f"    ... and {len(diffs) - 8} more")

    if ok and not structural_fail and not numeric_fail:
        scope = ("structure + claims" if args.cross_platform
                 else f"every number within rtol={args.rtol:g}/atol={args.atol:g}")
        print(f"\nOK: regenerated results match {args.ref} ({scope}, {len(files)} files).")

    if args.conclusions or args.cross_platform:
        broken = check_conclusions()
        if broken:
            ok = False
            print("\nFAIL: a CLAIM the project makes no longer holds:\n")
            for b in broken:
                print(f"    {b}")
        else:
            print("OK: every claim the project makes still holds on this platform.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
