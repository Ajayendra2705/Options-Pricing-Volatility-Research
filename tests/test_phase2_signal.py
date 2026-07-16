"""
Phase-2 (Day 33) — SPY RV/HAR/signal gate tests.
================================================
Day 32 built the arb-free SPY surface; this gates the signal built on top of it
by v1's own code (scripts/run_phase2_signal.py, config/spy_phase2.yaml). The
assertions hold Phase 2 to v1's conventions — expanding no-lookahead HAR,
raw-signal ranking, one short_vol + one long_vol per date — at 147-session
scale, plus the isolation guarantee that v1's tracked artifacts are untouched.

Skips wholesale if the Phase-2 signal has not been built.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P2_PROCESSED = PROJECT_ROOT / "data" / "phase2" / "processed"
P2_RESULTS = PROJECT_ROOT / "results" / "phase2"

SIGNAL = P2_PROCESSED / "signal.parquet"
HAR_STATS = P2_RESULTS / "har_stats_spy.json"
SIGNAL_SUMMARY = P2_RESULTS / "signal_summary_spy.json"

N_SESSIONS = 147

pytestmark = pytest.mark.skipif(
    not (SIGNAL.exists() and SIGNAL_SUMMARY.exists()),
    reason="Phase-2 SPY signal not built (run scripts/run_phase2_signal.py)",
)


@pytest.fixture(scope="module")
def sig() -> pd.DataFrame:
    return pd.read_parquet(SIGNAL)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(SIGNAL_SUMMARY.read_text())


# ── coverage: every session got a full signal cross-section ──────────────────

def test_signal_covers_every_session(sig):
    assert sig["date"].nunique() == N_SESSIONS
    assert sig["signal_raw"].notna().all()          # no slice missing a forecast


def test_one_short_one_long_per_date(sig):
    per = sig.groupby("date")["side"].value_counts().unstack(fill_value=0)
    assert (per["short_vol"] == 1).all()
    assert (per["long_vol"] == 1).all()


def test_ranking_matches_raw_signal(sig):
    # rank 1 (short_vol) must be the max raw signal of its date, last the min
    for date, g in sig.groupby("date"):
        assert g.loc[g["side"] == "short_vol", "signal_raw"].iloc[0] == g["signal_raw"].max()
        assert g.loc[g["side"] == "long_vol", "signal_raw"].iloc[0] == g["signal_raw"].min()


def test_horizon_clamped_to_option_life(sig):
    # SPY DTE 11-65 -> h = clamp(round(252*T), 5, 63); every h inside the clamp
    assert sig["h"].between(5, 63).all()
    expected = (252 * sig["T"]).round().clip(5, 63).astype(int)
    assert (sig["h"] == expected).all()


# ── the HAR leg: real out-of-sample forecasting power ────────────────────────

def test_har_oos_forecast_is_real():
    stats = json.loads(HAR_STATS.read_text())
    oos = stats["oos_expanding"]
    assert oos["n"] > 300                            # expanding column mostly filled
    assert oos["corr"] > 0.5, oos                    # forecasts track realized vol
    assert stats["r2_insample"] > 0.3


# ── study framing ────────────────────────────────────────────────────────────

def test_summary_names_the_phase2_prereg(summary):
    assert "spy_phase2.yaml" in summary["config"]
    assert summary["n_slices"] == summary["n_with_signal"]


def test_signal_is_centered_not_one_sided(summary):
    # 12 months of SPY: ATM IV ~ HAR forecast on average. If the mean signal
    # were as one-sided as v1's June-2023 snapshot (-4.7 volpts), the ranking
    # would be a level bet, not a cross-sectional one. Bound it loosely — this
    # documents the regime, it does not tune anything.
    assert abs(summary["signal_volpts"]["mean"]) < 2.0
    assert summary["signal_volpts"]["min"] < 0 < summary["signal_volpts"]["max"]


# ── isolation: v1 untouched ──────────────────────────────────────────────────

def test_v1_signal_untouched():
    v1 = json.loads((PROJECT_ROOT / "results" / "signal_summary.json").read_text())
    assert v1["config"] == "config/primary.yaml (pre-registered)"
    assert v1["n_slices"] == 15                      # AAPL: 5 dates x 3
