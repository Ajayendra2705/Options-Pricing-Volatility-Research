"""
Tests for the cross-platform reproducibility gate (scripts/compare_results.py).

The gate's whole job is to be tolerant of the float noise that a different
BLAS produces (~1e-9 relative) while catching anything that actually changed —
a moved number, a flipped flag, a renamed field, a Windows path baked into a
tracked artifact. Both halves are tested; a gate that never fails is not a gate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "compare_results", PROJECT_ROOT / "scripts" / "compare_results.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

RTOL, ATOL = 1e-6, 1e-12


def diffs(old, new, rtol=RTOL):
    return cr.compare(old, new, rtol, ATOL)


# ── tolerant where it should be ─────────────────────────────────────────────

def test_blas_scale_float_noise_is_not_a_diff():
    """~1e-9 relative drift (a different LAPACK landing a few ulps apart) passes."""
    old = {"sharpe": -1.7007935601148596, "beta": -1.5901234567,
           "nested": [{"g": 2.0000103e-4}]}
    new = {"sharpe": -1.7007935601148000, "beta": -1.5901234581,
           "nested": [{"g": 2.0000104e-4}]}
    assert diffs(old, new) == []


def test_nan_equals_nan():
    assert diffs({"x": float("nan")}, {"x": float("nan")}) == []


# ── strict where it should be ───────────────────────────────────────────────

def test_real_number_move_is_caught():
    # 1e-4 relative — far beyond BLAS noise, e.g. a changed model
    out = diffs({"sharpe": -1.70}, {"sharpe": -1.7002})
    assert len(out) == 1 and "sharpe" in out[0]


def test_windows_path_in_an_artifact_is_caught():
    out = diffs({"output": r"data\processed\chain_clean.parquet"},
                {"output": "data/processed/chain_clean.parquet"})
    assert len(out) == 1 and "output" in out[0]


def test_bools_are_never_tolerance_compared():
    """True == 1 numerically; a flipped arb-free flag must NOT slip through."""
    out = diffs({"all_interp_butterfly_ok": True},
                {"all_interp_butterfly_ok": False})
    assert len(out) == 1
    assert diffs({"ok": True}, {"ok": 1.0}) != []


def test_structure_changes_are_caught():
    assert diffs({"a": 1}, {"a": 1, "b": 2}) == ["/b: added"]
    assert diffs({"a": 1, "b": 2}, {"a": 1}) == ["/b: REMOVED"]
    assert diffs({"days": [1, 2, 3]}, {"days": [1, 2]}) != []


def test_nested_path_is_reported():
    out = diffs({"horizons": {"daily": {"sharpe": -1.70}}},
                {"horizons": {"daily": {"sharpe": -2.50}}})
    assert out == [f"/horizons/daily/sharpe: -1.7 -> -2.5 (rel 3.20e-01)"]


# ── real artifacts ──────────────────────────────────────────────────────────

def test_committed_results_match_the_working_tree():
    """The tracked JSONs are current: what the pipeline produces now equals what
    is committed (within tolerance). Fails if someone commits code without
    regenerating results."""
    import subprocess
    r = subprocess.run(["python", "scripts/compare_results.py", "--rtol", "1e-6"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
