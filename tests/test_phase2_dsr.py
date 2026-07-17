"""
Phase-2 Day 38 — Deflated Sharpe with honest N (v2 capstone) gate tests.
========================================================================
Two layers: (1) unit tests on the PSR/DSR math added to stats.py (inverse-normal
accuracy, PSR monotonicity, expected-max-Sharpe growth in N), and (2) the
artifact-honesty properties of scripts/run_phase2_dsr.py -- the trial ledger is
explicit and N matches its length, the deferred stub is now resolved
(computed=True), the headline cannot clear zero, and the best data-mined slice
does not clear the honest-N deflated bar.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.backtest.stats import (deflated_sharpe_ratio,
                                expected_max_sharpe_period, _norm_ppf,
                                _norm_cdf, probabilistic_sharpe_ratio)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DSR = PROJECT_ROOT / "results" / "phase2" / "deflated_sharpe_spy.json"


# ── unit tests on the math (always run) ──────────────────────────────────────

def test_norm_ppf_inverts_cdf():
    for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)


def test_norm_ppf_known_quantiles():
    assert _norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)


def test_psr_monotone_and_bounded():
    # higher observed Sharpe -> higher P(true > benchmark); always in (0,1)
    lo = probabilistic_sharpe_ratio(0.02, 0.0, 250, 0.0, 3.0)
    hi = probabilistic_sharpe_ratio(0.08, 0.0, 250, 0.0, 3.0)
    assert 0.0 < lo < hi < 1.0
    # a negative Sharpe cannot clear a positive benchmark
    assert probabilistic_sharpe_ratio(-0.05, 0.05, 250, 0.0, 3.0) < 0.5


def test_expected_max_sharpe_grows_with_n():
    v = 0.01
    bars = [expected_max_sharpe_period(v, n) for n in (2, 5, 20, 100)]
    assert bars == sorted(bars)          # more trials -> higher expected max
    assert expected_max_sharpe_period(v, 1) == 0.0


def test_deflated_sharpe_penalises_more_trials():
    # same observed Sharpe, more trials -> lower DSR (harder to be significant)
    kw = dict(sr_period=0.10, n_obs=250, skew=0.0, kurt=3.0,
              var_sr_trials_period=0.01)
    d5 = deflated_sharpe_ratio(**kw, n_trials=5)["dsr"]
    d50 = deflated_sharpe_ratio(**kw, n_trials=50)["dsr"]
    assert d5 > d50


# ── artifact honesty (skips if the DSR was not run) ──────────────────────────

pytestmark_artifact = pytest.mark.skipif(
    not DSR.exists(), reason="Phase-2 DSR not run (scripts/run_phase2_dsr.py)")


@pytest.fixture(scope="module")
def dsr() -> dict:
    return json.loads(DSR.read_text())


@pytestmark_artifact
def test_trial_ledger_is_explicit_and_n_matches(dsr):
    led = dsr["trial_ledger"]
    assert led["n_trials"] == len(led["trials"])
    assert led["n_trials"] >= 6                       # v1 + SPY + the sweeps
    # ledger carries v1 and the SPY sweeps, not just the headline
    names = " ".join(t["name"] for t in led["trials"])
    assert "v1" in names and "hedge every" in names and "regime" in names


@pytestmark_artifact
def test_deferral_is_now_resolved(dsr):
    ds = dsr["deflated_sharpe"]
    assert ds["computed"] is True
    assert "metrics_spy.json" in ds["supersedes"]
    assert ds["n_trials"] == dsr["trial_ledger"]["n_trials"]


@pytestmark_artifact
def test_headline_cannot_clear_zero(dsr):
    assert dsr["headline_book"]["psr_vs_zero"] < 0.5
    assert dsr["deflated_sharpe"]["dsr"] < 0.05


@pytestmark_artifact
def test_best_slice_does_not_survive_honest_deflation(dsr):
    ds = dsr["deflated_sharpe"]
    assert ds["best_data_mined_slice_survives_deflation"] is False
    best = dsr["trial_ledger"]["best_trial"]["sharpe_ann"]
    assert best < ds["expected_max_sharpe_ann"]       # +2.79 < +3.07


@pytestmark_artifact
def test_dsr_near_zero_across_all_n(dsr):
    for s in dsr["deflated_sharpe"]["sensitivity_over_n"]:
        assert s["dsr_headline"] < 0.05
        # the deflated bar rises with N (never easier to be significant)
    bars = [s["expected_max_sharpe_ann"]
            for s in dsr["deflated_sharpe"]["sensitivity_over_n"]]
    assert bars == sorted(bars)
