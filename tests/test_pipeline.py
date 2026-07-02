"""
Day 15 tests: Part-1 pipeline lock.

The full end-to-end rerun (python main.py --stage all) was verified
bit-identical against a SHA256 snapshot of all processed parquets + result
jsons; it is too slow for every pytest run. These tests lock the wiring and
cross-artifact consistency instead.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"


def test_stage_registry_and_runners_importable():
    # stage choices present in the CLI
    src = (PROJECT_ROOT / "main.py").read_text()
    for stage in ("clean", "surface", "backtest", "report"):
        assert f'"{stage}"' in src
    # every stage-1 runner used by main is importable and callable no-arg
    from src.surface.assemble import run_assembly
    from src.surface.clean import run_cleaning
    from src.surface.forwards import run_forwards
    from src.surface.iv_surface import run_iv_surface
    from src.surface.no_arb import run_arb_check
    from src.surface.svi import run_svi_all
    for fn in (run_cleaning, run_forwards, run_iv_surface, run_svi_all,
               run_arb_check, run_assembly):
        assert callable(fn)


@pytest.mark.skipif(not (PROCESSED / "svi_params_joint.parquet").exists(),
                    reason="pipeline not run")
def test_artifacts_cross_consistent():
    """Every stage's output agrees with the next stage's input expectations."""
    chain = pd.read_parquet(PROCESSED / "chain_clean.parquet")
    forwards = pd.read_parquet(PROCESSED / "forwards.parquet")
    surf = pd.read_parquet(PROCESSED / "iv_surface.parquet")
    joint = pd.read_parquet(PROCESSED / "svi_params_joint.parquet")

    # forwards cover exactly the (date, expiry) slices of the cleaned chain
    chain_slices = set(map(tuple, chain[["date", "expiry"]].drop_duplicates().values))
    fwd_slices = set(map(tuple, forwards[["date", "expiry"]].values))
    assert fwd_slices == chain_slices

    # iv surface rows = cleaned quotes; joint fits cover the same slices
    assert len(surf) == len(chain)
    joint_slices = set(map(tuple, joint[["date", "expiry"]].values))
    assert joint_slices == chain_slices

    # QC json agrees with the joint parquet
    qc = json.loads((PROJECT_ROOT / "results" / "surface_qc.json").read_text())
    ok = joint[joint["fit_ok"] == True]  # noqa: E712
    assert qc["n_slices_total"] == len(ok)
    assert qc["n_dates"] == joint["date"].nunique()

    # arb json agrees too
    arb = json.loads((PROJECT_ROOT / "results" / "arb_violations.json").read_text())
    assert arb["n_slices_fitted"] == len(ok)
    assert arb["butterfly"]["n_violations"] == 0
    assert arb["calendar"]["n_pairs_violated"] == 0
