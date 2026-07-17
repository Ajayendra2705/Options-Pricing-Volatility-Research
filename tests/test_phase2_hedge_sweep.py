"""
Phase-2 Day 37 — SPY hedge-frequency sweep (v2 robustness) gate tests.
======================================================================
Pins the sweep run by scripts/run_phase2_hedge_sweep.py: the daily point
(hedge_every=1) must reproduce the Day-34 returns and the Day-36 hedge notional
EXACTLY (validates the re-implemented aggregation), the cost/variance trade-off
must be monotone (turnover down, return-vol up as spacing grows), net PnL must
stay negative at every cadence, and the un-implemented band variant must be
disclosed, not faked. Nothing here touches a tracked artifact.

Skips wholesale if the sweep has not been run.
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P2_RESULTS = PROJECT_ROOT / "results" / "phase2"
SWEEP = P2_RESULTS / "hedge_sweep_spy.json"

pytestmark = pytest.mark.skipif(
    not SWEEP.exists(),
    reason="Phase-2 SPY hedge sweep not run (scripts/run_phase2_hedge_sweep.py)",
)


@pytest.fixture(scope="module")
def sweep() -> dict:
    return json.loads(SWEEP.read_text())


def _daily(sweep) -> dict:
    return next(c for c in sweep["curve"] if c["hedge_every_bars"] == 1)


def test_daily_point_reproduces_day34_returns(sweep):
    r = json.loads((P2_RESULTS / "returns_summary_spy.json").read_text())
    d = _daily(sweep)
    assert d["gross_pnl_usd"] == pytest.approx(r["gross_pnl_usd"], abs=1e-6)
    assert d["net_pnl_usd"] == pytest.approx(r["net_pnl_usd"], abs=1e-6)


def test_daily_turnover_reproduces_day36_notional(sweep):
    cs = json.loads((P2_RESULTS / "cost_sweep_spy.json").read_text())
    d = _daily(sweep)
    assert d["turnover_usd"] == pytest.approx(
        cs["hedge_slippage_stress"]["total_hedge_notional_usd"], rel=1e-9)


def test_turnover_falls_as_spacing_grows(sweep):
    curve = sorted(sweep["curve"], key=lambda c: c["hedge_every_bars"])
    tos = [c["turnover_usd"] for c in curve]
    assert tos == sorted(tos, reverse=True)          # strictly less trading


def test_return_vol_rises_as_spacing_grows(sweep):
    # the pre-registered "variance sensitivity": less hedging -> more variance
    curve = sorted(sweep["curve"], key=lambda c: c["hedge_every_bars"])
    vols = [c["return_vol_annualized"] for c in curve]
    assert vols == sorted(vols)


def test_no_cadence_makes_the_book_profitable(sweep):
    assert all(c["net_pnl_usd"] < 0 for c in sweep["curve"])
    # daily is the least-bad (short gamma bleeds more when under-hedged)
    best = max(sweep["curve"], key=lambda c: c["net_pnl_usd"])
    assert best["hedge_every_bars"] == 1


def test_band_hedging_is_disclosed_not_faked(sweep):
    band = sweep["band_hedging"]
    assert band["implemented"] is False
    assert "engine change" in band["reason"]


def test_capital_base_matches_day34(sweep):
    r = json.loads((P2_RESULTS / "returns_summary_spy.json").read_text())
    assert sweep["capital_base_usd"] == r["capital_base_usd"]
