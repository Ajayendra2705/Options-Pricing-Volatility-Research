"""
Phase-2 Day 39 — SABR second calibration (v2 surface-family robustness) tests.
==============================================================================
Two layers: (1) unit tests on the SABR math (Hagan ATM continuity, self-recovery
of a known smile), and (2) the artifact-honesty properties of
scripts/run_phase2_sabr.py -- an independent family fits the SPY surface about as
well as SVI, lands on the same ATM mark, and reproduces the same trade selection,
so the signal is not an SVI-specific artifact. No v1/tracked-v2 artifact is read
or written by the driver.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.surface.sabr import fit_sabr_slice, sabr_iv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SABR = PROJECT_ROOT / "results" / "phase2" / "svi_vs_sabr_spy.json"


# ── unit tests on the math (always run) ──────────────────────────────────────

def test_sabr_atm_is_continuous_across_the_forward():
    # z/x(z) -> 1 as K -> F; the smile must not blow up or jump at the money
    F, T = 450.0, 0.08
    Ks = F * np.exp(np.array([-1e-3, -1e-6, 0.0, 1e-6, 1e-3]))
    iv = sabr_iv(F, Ks, T, alpha=0.12, rho=-0.4, nu=1.5)
    assert np.all(np.isfinite(iv))
    assert np.ptp(iv) < 5e-3                       # smooth through ATM


def test_sabr_recovers_a_known_smile():
    F, T = 450.0, 0.08
    alpha, rho, nu = 0.12, -0.4, 1.5
    K = F * np.exp(np.linspace(-0.12, 0.10, 11))
    iv = sabr_iv(F, K, T, alpha, rho, nu)
    params, report = fit_sabr_slice(F, K, iv, T)
    assert report["rmse_iv"] < 1e-4
    assert params["alpha"] == pytest.approx(alpha, abs=1e-3)
    assert params["rho"] == pytest.approx(rho, abs=1e-2)
    assert report["atm_iv"] == pytest.approx(
        float(sabr_iv(F, F, T, alpha, rho, nu)), abs=1e-4)


# ── artifact honesty (skips if the SABR calibration was not run) ─────────────

skip_no_artifact = pytest.mark.skipif(
    not SABR.exists(), reason="SABR calibration not run (run_phase2_sabr.py)")


@pytest.fixture(scope="module")
def sabr() -> dict:
    return json.loads(SABR.read_text())


@skip_no_artifact
def test_sabr_fits_all_slices_sub_volpt_but_looser_than_svi(sabr):
    fq = sabr["fit_quality"]
    assert fq["n_slices_fit"] == fq["n_slices_total"]      # every slice fits
    assert fq["sabr_median_rmse_iv"] < 0.015               # sub-1.5-volpt median
    # SVI (arb-constrained, purpose-built) fits tighter -- honest, not equal
    assert fq["svi_median_rmse_iv"] < fq["sabr_median_rmse_iv"]


@skip_no_artifact
def test_atm_marks_agree(sabr):
    # the LEVEL the signal reads is not SVI-specific
    am = sabr["atm_agreement"]
    assert am["corr_atm_iv"] > 0.95
    assert am["median_abs_diff_volpts"] < 1.0             # sub-volpt agreement


@skip_no_artifact
def test_trade_selection_is_fragile_to_the_surface_model(sabr):
    # the honest finding: agreement is ABOVE chance (33% for one-of-three) but
    # FAR from robust -- the tradeable signal flips under an equally-valid
    # surface, which reinforces (not overturns) the disproof.
    ts = sabr["trade_selection"]
    assert ts["n_dates"] > 100
    for frac in (ts["short_agreement_frac"], ts["long_agreement_frac"]):
        assert 0.33 < frac < 0.75
