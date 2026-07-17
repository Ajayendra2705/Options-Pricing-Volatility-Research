"""
Phase-2 (Day 34) — SPY walk-forward backtest gate tests.
========================================================
Gates the full backtest chain run by scripts/run_phase2_backtest.py: the dual
attribution verdict (pre-registered FAIL disclosed + amended PASS), the
portfolio/costs/returns artifacts, the merged metrics, and the pre-registered
walk-forward fold report. Also pins the honesty properties: the pre-registered
gate's failure must stay ON THE RECORD, and v1's artifacts must be untouched.

Skips wholesale if the Phase-2 backtest has not been run.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P2_PROCESSED = PROJECT_ROOT / "data" / "phase2" / "processed"
P2_RESULTS = PROJECT_ROOT / "results" / "phase2"

GATE = P2_RESULTS / "attribution_gate_spy.json"
RECONCILE = P2_RESULTS / "attribution_reconcile_spy.json"
WALKFORWARD = P2_RESULTS / "walkforward_spy.json"

N_POSITIONS = 294               # 147 sessions x (1 short + 1 long)

pytestmark = pytest.mark.skipif(
    not (GATE.exists() and WALKFORWARD.exists()),
    reason="Phase-2 SPY backtest not run (scripts/run_phase2_backtest.py)",
)


@pytest.fixture(scope="module")
def gate() -> dict:
    return json.loads(GATE.read_text())


@pytest.fixture(scope="module")
def rec() -> dict:
    return json.loads(RECONCILE.read_text())


@pytest.fixture(scope="module")
def wf() -> dict:
    return json.loads(WALKFORWARD.read_text())


# ── the dual attribution verdict ─────────────────────────────────────────────

def test_preregistered_gate_failure_stays_on_the_record(gate):
    # the amendment must never erase the pre-registered verdict
    pre = gate["preregistered"]
    assert pre["pass"] is False
    assert pre["worst_position_residual_over_premium"] >= 0.10
    assert pre["book_residual_abs_share"] < 0.20        # book-level bar passed


def test_amended_gate_passed_and_is_documented(gate):
    am = gate["amended"]
    assert am["pass"] is True
    assert am["p95_residual_over_premium_ex_settlement"] < 0.10
    assert am["median_residual_over_premium"] < 0.05    # decomposition healthy
    assert "before portfolio" in am["amended_when"]
    assert "Taylor" in am["mechanism"]


def test_reconcile_covers_the_full_book(rec):
    assert rec["n_positions"] == N_POSITIONS
    assert all("residual_over_premium_ex_settlement" in p
               for p in rec["positions"])
    # sanity cap: no position, settlement included, above 0.50
    assert rec["worst_position_residual_over_premium"] < 0.50


# ── downstream artifacts exist and reconcile with each other ─────────────────

def test_costs_exceed_zero_and_net_below_gross():
    c = json.loads((P2_RESULTS / "costs_summary_spy.json").read_text())
    assert c["n_positions"] == N_POSITIONS
    assert c["total_cost"] > 0
    assert c["net_pnl"] == pytest.approx(c["gross_pnl"] - c["total_cost"])


def test_returns_capital_base_is_peak_margin():
    r = json.loads((P2_RESULTS / "returns_summary_spy.json").read_text())
    assert r["n_positions"] == N_POSITIONS
    assert r["capital_base_usd"] == r["peak_book_margin_usd"] > 0
    assert r["net_pnl_usd"] < r["gross_pnl_usd"]        # costs are real


def test_metrics_merged_with_honest_blocks():
    m = json.loads((P2_RESULTS / "metrics_spy.json").read_text())
    assert set(m["horizons"]) == {"daily", "weekly", "per_trade"}
    assert m["horizons"]["per_trade"]["n"] == N_POSITIONS
    sh = m["statistical_honesty"]
    assert sh["deflated_sharpe"]["computed"] is False   # deferred, not faked
    assert "Point Sharpe is" in sh["interpretation"]    # auto text, sign-aware


def test_no_fixed_name_views_left_behind():
    # the driver's temporary v1-named views must not survive in results/phase2
    for name in ("returns_summary.json", "costs_summary.json", "metrics.json"):
        assert not (P2_RESULTS / name).exists(), name


# ── the walk-forward folds ───────────────────────────────────────────────────

def test_folds_match_the_preregistration(wf):
    assert set(wf["folds"]) == {"2023Q4", "2024Q1", "2024Q2"}
    assert wf["burn_in"]["through"] == "2023-09-30"
    for f in wf["folds"].values():
        assert f["n_days"] > 50                          # a real quarter of days


def test_settlement_tail_is_reported_separately(wf):
    # positions run to expiry past the last fold; the run-off holds the
    # 2024-08-05 vol spike, so it must be its own labelled bucket
    tail = wf["settlement_tail"]
    assert tail["from"] == "2024-07-01"
    assert tail["n_days"] > 0
    assert "2024-08-05" in tail["note"]


def test_fold_pnl_plus_settlement_tail_sums_to_all_test_folds(wf):
    total = (sum(f["net_pnl_usd"] for f in wf["folds"].values())
             + wf["settlement_tail"]["net_pnl_usd"])
    assert total == pytest.approx(wf["all_test_folds"]["net_pnl_usd"], abs=1e-6)


def test_fold_report_uses_the_returns_capital_base(wf):
    r = json.loads((P2_RESULTS / "returns_summary_spy.json").read_text())
    assert wf["capital_base_usd"] == r["capital_base_usd"]


# ── isolation: v1 untouched ──────────────────────────────────────────────────

def test_v1_reconcile_untouched():
    v1 = json.loads((PROJECT_ROOT / "results" /
                     "attribution_reconcile.json").read_text())
    assert v1["n_positions"] == 10                       # AAPL book
    assert "p95_residual_over_premium_ex_settlement" not in v1


def test_v1_metrics_untouched():
    v1 = json.loads((PROJECT_ROOT / "results" / "metrics.json").read_text())
    assert "n=27/side" in v1["statistical_honesty"]["interpretation"]
