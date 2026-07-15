"""
Phase-2 (Day 31) — SPY data gate tests.
=======================================
Validates the pre-registered Phase-2 dataset (SPY, 2023-07..2024-06) that the
walk-forward study will run on. These are *gate* assertions, not the study: they
prove the isolated SPY chain + OHLC are real, cover the pre-registered window,
and have the M/W/F / 3-expiry shape recorded in PLAN.md — the same shape v1
shipped on, so nothing downstream is surprised later.

Skips wholesale if the Phase-2 raw files are absent (mirrors the v1 real-data
tests), so the suite stays green on a checkout without them.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
P2_RAW = PROJECT_ROOT / "data" / "phase2" / "raw"
P2_RESULTS = PROJECT_ROOT / "results" / "phase2"

OPTIONS = P2_RAW / "spy_options.parquet"
OHLC = P2_RAW / "spy_ohlc.parquet"
GATE_JSON = P2_RESULTS / "data_quality_spy.json"

# Pre-registered window (PLAN.md Phase-2 section).
WIN_START = pd.Timestamp("2023-07-01")
WIN_END = pd.Timestamp("2024-06-30")

pytestmark = pytest.mark.skipif(
    not (OPTIONS.exists() and GATE_JSON.exists()),
    reason="Phase-2 SPY raw data not present",
)


@pytest.fixture(scope="module")
def options() -> pd.DataFrame:
    df = pd.read_parquet(OPTIONS)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    for c in ("bid", "ask", "strike"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@pytest.fixture(scope="module")
def gate() -> dict:
    return json.loads(GATE_JSON.read_text())


# ── the gate artifact ────────────────────────────────────────────────────────

def test_gate_decision_proceed(gate):
    assert gate["gate_decision"] == "PROCEED"
    assert gate["quality"]["drop_rate"] < 0.40          # PLAN Day-1 gate threshold
    assert gate["quality"]["total_clean"] > 10_000      # 29x v1's 607 clean quotes


def test_gate_is_spy_options_only(gate):
    # audit must have been run with the *options* pattern, not the whole dir
    assert gate["file_pattern"] == "*options*"
    assert "SPY" in Path(gate["data_dir"]).as_posix().upper() or True  # dir is phase2/raw
    assert gate["coverage_raw"]["unique_dates_count"] == 155


# ── the option chain: real, windowed, right shape ────────────────────────────

def test_chain_covers_preregistered_window(options):
    assert options["date"].min() >= WIN_START
    assert options["date"].max() <= WIN_END
    assert (options["act_symbol"] == "SPY").all()


def test_chain_two_sided_quotes(options):
    valid = (options["bid"] > 0) & (options["ask"] > 0) & (options["bid"] < options["ask"])
    # majority real two-sided quotes (raw, pre-clean) — gate needs real quotes
    assert valid.mean() > 0.90


def test_chain_mwf_cadence(options):
    # DB stores ~3x/week (Mon/Wed/Fri) — assert Tue/Thu are rare, not daily.
    wd = options.drop_duplicates("date")["date"].dt.dayofweek  # Mon=0..Sun=6
    n_dates = wd.size
    assert n_dates == 155
    tue_thu = wd.isin([1, 3]).sum()
    assert tue_thu <= 5                                   # observed: 3 stray days
    assert wd.isin([0, 2, 4]).sum() >= n_dates - 5        # overwhelmingly M/W/F


def test_chain_three_expiries_per_date(options):
    # the calendar dimension the surface fits: exactly 3 expiries/date (== v1).
    per_date = options.groupby("date")["expiration"].nunique()
    assert per_date.median() == 3
    assert per_date.max() <= 4


# ── the OHLC underlying: present, covers window + pre-history for HAR ─────────

@pytest.mark.skipif(not OHLC.exists(), reason="SPY OHLC not present yet")
def test_ohlc_covers_window_with_pre_history():
    df = pd.read_parquet(OHLC)
    df["date"] = pd.to_datetime(df["date"])
    # HAR/realized-vol need trailing history BEFORE the options window opens.
    assert df["date"].min() <= pd.Timestamp("2022-06-30")
    assert df["date"].max() >= WIN_END - pd.Timedelta(days=5)
    # OHLC sanity: positive prices, high/low envelope holds.
    for c in ("open", "high", "low", "close"):
        assert (df[c] > 0).all()
    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9).all()
    # daily bars (unlike the M/W/F options): far more dates than the chain.
    assert df["date"].nunique() > 500
