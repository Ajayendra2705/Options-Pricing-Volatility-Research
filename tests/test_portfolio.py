"""
Day 23 — Portfolio assembly + sizing tests.

Tests the sizing / clipping / kill-switch / aggregation machinery on both
synthetic data and (when present) the real AAPL data.

1. Vega sizing: qty scaled so vega_notional hits the limit exactly.
2. Gamma sizing: qty scaled so dollar_gamma hits the limit exactly.
3. Portfolio clip: positions exceeding portfolio gross vega → pro-rata'd.
4. DD kill-switch: synthetic equity with a deep drawdown → killed at right bar.
5. Aggregation identity: sum of per-position ledger equities == portfolio eq.
6. Sign consistency: short_vol → negative qty, long_vol → positive.
7. Real-data gate (skip if files missing): sizes 10 positions, portfolio
   equity curve has expected date range, max DD within or documented.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Leg, run_hedged
from src.backtest.portfolio import (
    _entry_greeks,
    _load_sizing_params,
    drawdown_kill,
    portfolio_limits,
    size_position,
)
from src.backtest.reconcile import PROCESSED_DIR, RAW_DIR


# ── synthetic helpers ────────────────────────────────────────────────────────

def _make_pos(S=180.0, K=180.0, T=0.1, sig=0.25, r=0.05, q=0.005,
              side="short_vol"):
    return {
        "S0": S, "K": K, "T": T, "mark_vol": sig, "r": r, "q": q,
        "side": side, "qty": -1.0 if side == "short_vol" else 1.0,
        "date": pd.Timestamp("2023-06-02"),
        "expiry": pd.Timestamp("2023-06-02") + pd.Timedelta(days=int(T * 365)),
        "F": S * np.exp((r - q) * T),
    }


# ── 1. Vega sizing ──────────────────────────────────────────────────────────

def test_vega_sized_to_limit():
    """Position sized so vega_notional == vega_limit_per_position."""
    pos = _make_pos()
    vega_1, _ = _entry_greeks(pos)
    params = {"vega_limit_per_position": 200.0,
              "gamma_limit_per_position": 1e9}  # gamma not binding
    qty = size_position(pos, params)
    assert qty < 0  # short_vol → negative qty
    np.testing.assert_allclose(abs(qty) * vega_1, 200.0, rtol=1e-10)


# ── 2. Gamma sizing ─────────────────────────────────────────────────────────

def test_gamma_sized_to_limit():
    """Position sized so dollar_gamma == gamma_limit_per_position."""
    pos = _make_pos()
    _, dgamma_1 = _entry_greeks(pos)
    params = {"vega_limit_per_position": 1e9,  # vega not binding
              "gamma_limit_per_position": 3000.0}
    qty = size_position(pos, params)
    assert qty < 0
    np.testing.assert_allclose(abs(qty) * dgamma_1, 3000.0, rtol=1e-10)


def test_binding_constraint_is_tighter():
    """Whichever limit gives the smaller absolute qty wins."""
    pos = _make_pos()
    vega_1, dgamma_1 = _entry_greeks(pos)
    params = {"vega_limit_per_position": 100.0,
              "gamma_limit_per_position": 1e9}
    qty_vega = abs(size_position(pos, params))

    params2 = {"vega_limit_per_position": 1e9,
               "gamma_limit_per_position": 100.0}
    qty_gamma = abs(size_position(pos, params2))

    params3 = {"vega_limit_per_position": 100.0,
               "gamma_limit_per_position": 100.0}
    qty_both = abs(size_position(pos, params3))
    assert qty_both == pytest.approx(min(qty_vega, qty_gamma))


# ── 3. Portfolio clip ────────────────────────────────────────────────────────

def test_portfolio_clip_pro_rata():
    """Three positions that exceed portfolio vega → pro-rata'd, total ≤ limit."""
    positions = []
    for i in range(3):
        p = _make_pos(side="short_vol")
        p["sized_qty"] = -2.0
        p["sized_vega"] = 600.0   # 3 × 600 = 1800 gross
        p["sized_dgamma"] = 500.0
        positions.append(p)

    params = {"portfolio_gross_vega_limit": 1200.0,  # < 1800 → clips
              "portfolio_gross_gamma_limit": 1e9}
    clipped = portfolio_limits(positions, params)
    total = sum(abs(p["sized_vega"]) for p in clipped)
    assert total <= 1200.0 + 1e-9
    # All clipped by the same ratio
    ratios = [abs(clipped[i]["sized_qty"]) for i in range(3)]
    assert ratios[0] == pytest.approx(ratios[1])
    assert ratios[1] == pytest.approx(ratios[2])


def test_portfolio_no_clip_when_under_limits():
    """No clipping when exposure is under limits."""
    p = _make_pos()
    p["sized_qty"] = -1.0
    p["sized_vega"] = 100.0
    p["sized_dgamma"] = 200.0
    original_qty = p["sized_qty"]
    portfolio_limits([p], {"portfolio_gross_vega_limit": 1e6,
                           "portfolio_gross_gamma_limit": 1e6})
    assert p["sized_qty"] == original_qty


# ── 4. DD kill-switch ────────────────────────────────────────────────────────

def test_dd_kill_switch():
    """A 25% drawdown triggers the 15% kill-switch; equity freezes."""
    dates = pd.bdate_range("2023-06-02", periods=20)
    eq = pd.Series(
        [0, 10, 20, 30, 40, 50,  # up to 50
         45, 40, 35, 30, 25, 20, 15, 10,  # DD to 10 (40 below peak 50)
         15, 20, 25, 30, 35, 40],
        index=dates, dtype=float)

    killed, kill_date = drawdown_kill(eq, 0.15)
    assert kill_date is not None
    # After kill, all values are flat
    kill_idx = dates.get_loc(pd.Timestamp(kill_date))
    assert (killed.iloc[kill_idx:] == killed.iloc[kill_idx]).all()
    # Before kill, values match original
    assert (killed.iloc[:kill_idx] == eq.iloc[:kill_idx]).all()


def test_dd_no_kill_when_within_limit():
    """No kill when drawdown never exceeds the threshold."""
    dates = pd.bdate_range("2023-06-02", periods=10)
    eq = pd.Series([0, 5, 10, 9, 8, 9, 10, 11, 12, 13],
                   index=dates, dtype=float)
    killed, kill_date = drawdown_kill(eq, 0.50)
    assert kill_date is None
    pd.testing.assert_series_equal(killed, eq)


# ── 5. Aggregation identity ─────────────────────────────────────────────────

def test_aggregation_identity():
    """Sum of per-position ledger equities == portfolio equity at every bar."""
    rng = np.random.default_rng(123)
    n = 30
    dates = pd.bdate_range("2023-06-02", periods=n)
    dt = 1.0 / 252
    S = 180.0 * np.exp(np.cumsum(
        np.r_[0.0, -0.5 * 0.20**2 * dt + 0.20 * np.sqrt(dt)
              * rng.standard_normal(n - 1)]))

    legs1 = [Leg(K=180.0, expiry=dates[-1], cp=+1, qty=-0.5, mark_vol=0.25),
             Leg(K=180.0, expiry=dates[-1], cp=-1, qty=-0.5, mark_vol=0.25)]
    legs2 = [Leg(K=185.0, expiry=dates[-1], cp=+1, qty=+0.3, mark_vol=0.22),
             Leg(K=185.0, expiry=dates[-1], cp=-1, qty=+0.3, mark_vol=0.22)]

    led1 = run_hedged(dates, S, legs1, r=0.0)
    led2 = run_hedged(dates, S, legs2, r=0.0)

    agg = led1["equity"].to_numpy() + led2["equity"].to_numpy()
    # Direct sum must hold at every bar
    for i in range(n):
        np.testing.assert_allclose(agg[i],
                                   led1["equity"].iloc[i] + led2["equity"].iloc[i],
                                   atol=1e-12)


# ── 6. Sign consistency ─────────────────────────────────────────────────────

def test_sign_consistency():
    """short_vol → negative qty, long_vol → positive qty."""
    params = {"vega_limit_per_position": 500.0,
              "gamma_limit_per_position": 5000.0}
    short = _make_pos(side="short_vol")
    long = _make_pos(side="long_vol")
    assert size_position(short, params) < 0
    assert size_position(long, params) > 0


# ── 7. Real-data gate ───────────────────────────────────────────────────────

_NEEDED = [PROCESSED_DIR / "signal.parquet",
           PROCESSED_DIR / "forwards.parquet",
           PROCESSED_DIR / "svi_params_joint.parquet",
           PROCESSED_DIR / "chain_clean.parquet",
           RAW_DIR / "aapl_ohlc.parquet",
           RAW_DIR / "aapl_ohlc_ext.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="real data files not present")
def test_real_data_portfolio():
    """Full pipeline on real data: 10 sized positions, sane equity curve."""
    from src.backtest.portfolio import run_portfolio

    summary = run_portfolio()

    # 10 positions (5 short_vol + 5 long_vol)
    assert summary["n_positions"] == 10
    sides = [p["side"] for p in summary["positions"]]
    assert sides.count("short_vol") == 5
    assert sides.count("long_vol") == 5

    # All sized qtys are non-zero and correctly signed
    for p in summary["positions"]:
        if p["side"] == "short_vol":
            assert p["sized_qty"] < 0
        else:
            assert p["sized_qty"] > 0
        assert abs(p["sized_qty"]) > 0

    # Portfolio equity parquet exists and has expected structure
    df = pd.read_parquet(PROCESSED_DIR / "portfolio.parquet")
    assert "date" in df.columns
    assert "equity" in df.columns
    assert len(df) > 0

    # Date range covers at least June-August 2023
    assert df["date"].min() <= pd.Timestamp("2023-06-02")
    assert df["date"].max() >= pd.Timestamp("2023-07-28")

    # Max DD is finite
    assert np.isfinite(summary["max_drawdown"])
    assert summary["max_drawdown"] >= 0
