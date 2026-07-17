"""
Phase-2 Day 36 — SPY cost sensitivity sweep (v2 robustness) gate tests.
=======================================================================
Pins the sweep run by scripts/run_phase2_cost_sweep.py: the x1 point of the
cost-multiplier curve must reproduce the actual Day-34 costs run EXACTLY (the
sweep is the true x k cost model, not a linearisation), the curve must be
monotone in the multiplier, the break-even multiplier must be negative (the
book loses at zero cost -> not a cost artifact), and the hedge-slippage stress
must be linear in bps and disclose the hedge turnover. Nothing here touches a
v1 or the tracked v2 costs artifact.

Skips wholesale if the sweep has not been run.
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P2_RESULTS = PROJECT_ROOT / "results" / "phase2"
SWEEP = P2_RESULTS / "cost_sweep_spy.json"

pytestmark = pytest.mark.skipif(
    not SWEEP.exists(),
    reason="Phase-2 SPY cost sweep not run (scripts/run_phase2_cost_sweep.py)",
)


@pytest.fixture(scope="module")
def sweep() -> dict:
    return json.loads(SWEEP.read_text())


def test_x1_reproduces_the_actual_costs_run(sweep):
    # the x1 point must equal the real Day-34 net, or the "exact" claim is false
    base = json.loads((P2_RESULTS / "costs_summary_spy.json").read_text())
    x1 = next(c for c in sweep["cost_multiplier_sweep"]["curve"]
              if c["multiplier"] == 1.0)
    assert x1["net_pnl_usd"] == pytest.approx(base["net_pnl"], abs=1e-6)
    x0 = next(c for c in sweep["cost_multiplier_sweep"]["curve"]
              if c["multiplier"] == 0.0)
    assert x0["net_pnl_usd"] == pytest.approx(base["gross_pnl"], abs=1e-6)


def test_cost_curve_is_monotone_decreasing(sweep):
    curve = sweep["cost_multiplier_sweep"]["curve"]
    ks = [c["multiplier"] for c in curve]
    assert ks == sorted(ks)
    nets = [c["net_pnl_usd"] for c in curve]
    assert nets == sorted(nets, reverse=True)     # more cost -> less net


def test_loses_at_zero_cost_so_not_a_cost_artifact(sweep):
    ms = sweep["cost_multiplier_sweep"]
    # gross already negative -> break-even multiplier is negative
    assert ms["break_even_multiplier"] < 0
    x0 = next(c for c in ms["curve"] if c["multiplier"] == 0.0)
    assert x0["net_pnl_usd"] < 0                    # loses even at zero cost


def test_hedge_slippage_is_linear_and_turnover_disclosed(sweep):
    hs = sweep["hedge_slippage_stress"]
    assert hs["total_hedge_notional_usd"] > 0
    # turnover is large relative to capital -- the real cost-sensitive axis
    assert hs["hedge_notional_over_capital"] > 1.0
    curve = {c["underlying_slippage_bps"]: c for c in hs["curve"]}
    notional = hs["total_hedge_notional_usd"]
    for bps, c in curve.items():
        assert c["hedge_slippage_usd"] == pytest.approx(bps / 1e4 * notional,
                                                        rel=1e-9)
    assert curve[0.0]["net_pnl_usd"] < 0            # negative before any slippage


def test_sweep_did_not_clobber_the_base_costs_artifact(sweep):
    # the 1-bp re-run must write only to scratch; tracked base stays at bps=0
    base = json.loads((P2_RESULTS / "costs_summary_spy.json").read_text())
    assert base["cost_params"]["underlying_slippage_bps"] == 0.0
    assert base["total_hedge_slippage"] == 0.0
