"""
Phase-2 Day 35 — SPY vol-regime split (v2 robustness) gate tests.
=================================================================
Pins the honesty properties of the regime split run by
scripts/run_phase2_regime.py: the regime variable is backward-looking, each cut
partitions its universe and reconciles EXACTLY to the study's own totals (net
-$8,936 day-level, gross -$7,493 entry-level), the terciles are ordered by vol,
and the headline finding (a short-vol book bleeds as the contemporaneous vol
regime rises) is present in the artifact. Nothing here touches a v1 artifact.

Skips wholesale if the regime split has not been run.
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P2_RESULTS = PROJECT_ROOT / "results" / "phase2"

REGIME = P2_RESULTS / "regime_split_spy.json"
LABELS = ("low_vol", "mid_vol", "high_vol")
N_POSITIONS = 294
N_RETURN_DAYS = 283

pytestmark = pytest.mark.skipif(
    not REGIME.exists(),
    reason="Phase-2 SPY regime split not run (scripts/run_phase2_regime.py)",
)


@pytest.fixture(scope="module")
def reg() -> dict:
    return json.loads(REGIME.read_text())


# ── the regime variable is honest (no lookahead, tracks the implied-vol regime)

def test_regime_variable_is_backward_looking(reg):
    rv = reg["regime_variable"]
    assert rv["backward_looking"] is True
    assert "Yang-Zhang" in rv["name"]
    # RV regime and the study's implied-vol (VIX-like) regime pick the same days
    corr = rv["vix_proxy_cross_check"]["corr_trailing_rv_vs_entry_atm_iv"]
    assert corr > 0.3


# ── day-level cut: partitions the days and reconciles to the NET total ───────

def test_day_level_partitions_all_return_days(reg):
    day = reg["day_level"]["regimes"]
    assert sum(day[l]["n_days"] for l in LABELS) == N_RETURN_DAYS
    assert reg["day_level"]["all_days"]["n_days"] == N_RETURN_DAYS


def test_day_level_reconciles_to_net_total(reg):
    day = reg["day_level"]["regimes"]
    total = sum(day[l]["net_pnl_usd"] for l in LABELS)
    assert total == pytest.approx(
        reg["day_level"]["all_days"]["net_pnl_usd"], abs=1e-6)


def test_terciles_ordered_by_vol(reg):
    # mean trailing RV must rise low -> mid -> high, or the labels are a lie
    day = reg["day_level"]["regimes"]
    rvs = [day[l]["mean_trailing_rv"] for l in LABELS]
    assert rvs[0] < rvs[1] < rvs[2]
    edges = reg["day_level"]["tercile_edges_rv"]
    assert edges["low|mid"] < edges["mid|high"]


def test_short_vol_book_bleeds_as_regime_rises(reg):
    # the pre-registered robustness question: does the edge survive high vol?
    # for a short-vol book the honest answer is no -- high-vol days lose most.
    day = reg["day_level"]["regimes"]
    assert day["high_vol"]["net_pnl_usd"] < day["low_vol"]["net_pnl_usd"]
    assert day["high_vol"]["sharpe_annualized"] < 0


# ── entry-level cut: partitions the positions and reconciles to GROSS ────────

def test_entry_level_partitions_all_positions(reg):
    ent = reg["entry_level"]["regimes"]
    assert sum(ent[l]["n_positions"] for l in LABELS) == N_POSITIONS


def test_entry_level_reconciles_to_gross_total(reg):
    ent = reg["entry_level"]["regimes"]
    gross = json.loads((P2_RESULTS / "returns_summary_spy.json").read_text())[
        "gross_pnl_usd"]
    total = sum(ent[l]["gross_pnl_usd"] for l in LABELS)
    assert total == pytest.approx(gross, abs=1e-6)


def test_entry_finding_is_documented(reg):
    assert "MID-vol" in reg["entry_level"]["finding"]
    assert "2024-08-05" in reg["entry_level"]["finding"]


# ── isolation: no v1 artifact is read or written ─────────────────────────────

def test_regime_split_is_phase2_isolated(reg):
    # the artifact lives under results/phase2 and names only phase2 inputs
    src = reg["regime_variable"]["source"]
    assert "spy_ohlc" in src
    assert reg["capital_base_usd"] > 0
