"""
Day 22 — THE reconciliation gate (PLAN: "validates the whole project").

Real data end-to-end: pre-registered book (one ATM straddle per non-flat
signal row) -> engine on real AAPL closes -> Day-21 Greeks attribution ->
residual must close.

What "closes" means here, with the Day-21 identity already proving the
residual is pure per-leg one-day Taylor error (no accounting leak is even
possible — hedge and financing terms are exact):
- book-wide, sum |residual| is a small share of sum |daily PnL| moves;
- no position's cumulative residual is more than a small share of its
  premium (worst observed: the Aug-18 long straddle across the Aug-3
  earnings gap, where a one-day expansion genuinely under-counts a long
  gamma gain — third-order, not a leak; sign and location both physical).

Thresholds are set with headroom above the observed values (0.144 share,
7.3% worst premium share) but tight enough that a real accounting leak —
which shows up as O(premium) drift, not O(dS^3) noise — trips the gate.

Skipped wholesale if the real data files are absent.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.attribution import TERM_COLS, leg_attribution
from src.backtest.engine import run_hedged
from src.backtest.reconcile import (PROCESSED_DIR, RAW_DIR, build_positions,
                                    load_price_path, run_position,
                                    run_reconcile)

_NEEDED = [PROCESSED_DIR / "signal.parquet", PROCESSED_DIR / "forwards.parquet",
           PROCESSED_DIR / "svi_params_joint.parquet",
           PROCESSED_DIR / "chain_clean.parquet",
           RAW_DIR / "aapl_ohlc.parquet", RAW_DIR / "aapl_ohlc_ext.parquet"]

pytestmark = pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                                reason="real data files not present")

MAX_BOOK_RESIDUAL_ABS_SHARE = 0.20
MAX_POSITION_RESIDUAL_OVER_PREMIUM = 0.10


@pytest.fixture(scope="module")
def positions():
    return build_positions()


@pytest.fixture(scope="module")
def report():
    return run_reconcile()


def test_book_matches_preregistration(positions):
    # primary.yaml: rank 1 short_vol + rank last long_vol per quote date,
    # 5 quote dates -> 10 positions; strike nearest listed to the forward
    assert len(positions) == 10
    sides = pd.Series([p["side"] for p in positions]).value_counts()
    assert sides["short_vol"] == 5 and sides["long_vol"] == 5
    for p in positions:
        assert abs(np.log(p["K"] / p["F"])) < 0.03      # ATM by construction
        assert 0.0 < p["mark_vol"] < 1.0
        # implied carry is sane for a mild dividend payer (backed out of F)
        assert -0.02 < p["q"] < 0.06


def test_engine_identity_on_real_paths(positions):
    path = load_price_path()
    for pos in positions:
        led, _, _ = run_position(pos, path)
        np.testing.assert_allclose(
            led["equity"], led["cash"] + led["shares"] * led["S"] + led["V_opt"],
            atol=1e-9)
        assert led["shares"].iloc[-1] == 0.0 and led["V_opt"].iloc[-1] == 0.0


def test_residual_is_pure_taylor_error_on_real_data(positions):
    # real-data rerun of the Day-21 identity: book residual == sum of
    # per-leg (mark move - Taylor terms); any accounting leak breaks this
    path = load_price_path()
    pos = positions[0]
    win = path[(path["date"] >= pos["date"]) & (path["date"] <= pos["expiry"])]
    _, book, legs = run_position(pos, path)
    led, legs_led = run_hedged(win["date"], win["close"].to_numpy(), legs,
                               r=pos["r"], q=pos["q"], return_legs=True)
    la = leg_attribution(win["date"], win["close"].to_numpy(), legs,
                         r=pos["r"], q=pos["q"])
    idx = pd.Index(led["date"])
    opt = legs_led.groupby("date")["pnl_day"].sum().reindex(idx, fill_value=0.0)
    terms = (la.groupby("date")[TERM_COLS].sum().sum(axis=1)
             .reindex(idx, fill_value=0.0))
    np.testing.assert_allclose(book["residual"], (opt - terms).to_numpy(),
                               atol=1e-9)


def test_reconciliation_gate(report):
    # THE gate. Residual closes book-wide and per position.
    assert report["n_positions"] == 10
    assert report["book_residual_abs_share"] < MAX_BOOK_RESIDUAL_ABS_SHARE
    assert (report["worst_position_residual_over_premium"]
            < MAX_POSITION_RESIDUAL_OVER_PREMIUM)
    for p in report["positions"]:
        # a leak would drift O(premium); Taylor noise stays well under it
        assert p["residual_abs_sum"] < 0.35 * p["premium"]


def test_vol_terms_zero_under_constant_marks(positions):
    # engine marks constant per-leg IV -> the decomposition's vol terms are
    # structurally zero on real data too (they activate on re-marking later)
    path = load_price_path()
    _, book, _ = run_position(positions[0], path)
    for c in ("vega_pnl", "vanna_pnl", "volga_pnl", "rho_pnl"):
        assert (book[c] == 0.0).all()
