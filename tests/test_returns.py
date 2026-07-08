"""
Day 25 — Margin-based returns tests.

1. Naked leg margin formula: max(0.20*S - OTM, 0.10*S) + premium.
2. Floor binds deep-OTM: requirement never below 0.10*S + premium.
3. Short straddle margin = larger naked leg + other leg's premium.
4. Long straddle margin = premium debit (call+put), fully paid.
5. Settled leg (T<=0) has zero margin.
6. Procyclicality direction: a short straddle's margin rises when the
   underlying moves away from the strike (spot stress -> more margin).
7. Real-data gate (skip if files missing): capital base = peak book margin,
   net return = net PnL / capital base, denominator documented, sane ranges.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.returns import (
    _naked_leg_margin,
    straddle_margin,
    REG_T_RATE,
    REG_T_FLOOR,
    MULT,
)
from src.backtest.reconcile import PROCESSED_DIR, RAW_DIR
from src.greeks.black_scholes import price_spot


# ── 1-2. naked leg margin ────────────────────────────────────────────────────

def test_naked_leg_margin_formula():
    S, K, T, sig, r, q = 180.0, 180.0, 0.1, 0.25, 0.05, 0.0
    req, prem = _naked_leg_margin(S, K, T, sig, r, q, +1)
    otm = max(K - S, 0.0)     # ATM call -> 0
    expect = max(REG_T_RATE * S - otm, REG_T_FLOOR * S) + prem
    np.testing.assert_allclose(req, expect, rtol=1e-12)
    # premium matches the pricer
    np.testing.assert_allclose(
        prem, float(price_spot(S, K, T, sig, r, q, +1)), rtol=1e-12)


def test_naked_margin_floor_binds_deep_otm():
    """Deep-OTM: 0.20*S - OTM would go below the 0.10*S floor -> floor holds."""
    S, K, T, sig, r, q = 180.0, 300.0, 0.1, 0.25, 0.05, 0.0
    req, prem = _naked_leg_margin(S, K, T, sig, r, q, +1)  # call, very OTM
    # 0.20*180 - (300-180) = 36 - 120 < 0 -> floor 0.10*180 = 18
    np.testing.assert_allclose(req - prem, REG_T_FLOOR * S, rtol=1e-12)


# ── 3. short straddle rule ───────────────────────────────────────────────────

def test_short_straddle_larger_leg_plus_other_premium():
    S, K, T, sig, r, q = 180.0, 175.0, 0.1, 0.25, 0.05, 0.0
    req_c, prem_c = _naked_leg_margin(S, K, T, sig, r, q, +1)
    req_p, prem_p = _naked_leg_margin(S, K, T, sig, r, q, -1)
    expect = (max(req_c, req_p) + (prem_p if req_c >= req_p else prem_c)) * MULT
    got = straddle_margin(S, K, T, sig, r, q, qty=-1.0)
    np.testing.assert_allclose(got, expect, rtol=1e-12)


# ── 4. long straddle = premium debit ─────────────────────────────────────────

def test_long_straddle_is_premium_debit():
    S, K, T, sig, r, q = 180.0, 180.0, 0.1, 0.25, 0.05, 0.0
    prem_c = float(price_spot(S, K, T, sig, r, q, +1))
    prem_p = float(price_spot(S, K, T, sig, r, q, -1))
    got = straddle_margin(S, K, T, sig, r, q, qty=+1.0)
    np.testing.assert_allclose(got, (prem_c + prem_p) * MULT, rtol=1e-12)


# ── 5. settled leg ───────────────────────────────────────────────────────────

def test_settled_leg_zero_margin():
    assert straddle_margin(180.0, 180.0, 0.0, 0.25, 0.05, 0.0, -1.0) == 0.0
    assert straddle_margin(180.0, 180.0, -0.1, 0.25, 0.05, 0.0, +1.0) == 0.0


# ── 6. procyclicality direction ──────────────────────────────────────────────

def test_short_straddle_margin_rises_with_spot_stress():
    """Moving spot away from the strike raises a short straddle's margin."""
    K, T, sig, r, q = 180.0, 0.1, 0.25, 0.05, 0.0
    m_atm = straddle_margin(180.0, K, T, sig, r, q, -1.0)
    m_up = straddle_margin(210.0, K, T, sig, r, q, -1.0)   # +30 move
    m_dn = straddle_margin(150.0, K, T, sig, r, q, -1.0)   # -30 move
    assert m_up > m_atm
    assert m_dn > m_atm


def test_qty_scales_margin():
    args = (180.0, 180.0, 0.1, 0.25, 0.05, 0.0)
    m1 = straddle_margin(*args, -1.0)
    m3 = straddle_margin(*args, -3.0)
    np.testing.assert_allclose(m3, 3.0 * m1, rtol=1e-12)


# ── 7. real-data gate ────────────────────────────────────────────────────────

_NEEDED = [PROCESSED_DIR / "signal.parquet",
           PROCESSED_DIR / "forwards.parquet",
           PROCESSED_DIR / "svi_params_joint.parquet",
           PROCESSED_DIR / "chain_clean.parquet",
           RAW_DIR / "aapl_ohlc.parquet",
           RAW_DIR / "aapl_ohlc_ext.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="real data files not present")
def test_real_data_returns():
    from src.backtest.returns import run_returns

    s = run_returns()

    assert s["n_positions"] == 10
    # capital base is the peak margin and >= entry margin
    assert s["capital_base_usd"] == pytest.approx(s["peak_book_margin_usd"])
    assert s["peak_book_margin_usd"] >= s["entry_book_margin_usd"] > 0
    # genuine (fixed-book) procyclicality: single-straddle margin expands
    # under spot moves, so the stress ratio exceeds 1
    assert s["per_position_margin_stress_ratio_max"] >= 1.0
    assert s["per_position_margin_stress_ratio_mean"] >= 1.0

    # net return = net PnL / capital base, exactly
    np.testing.assert_allclose(
        s["net_return_on_capital"],
        s["net_pnl_usd"] / s["capital_base_usd"], rtol=1e-9)
    # net PnL matches Day-24 (gross - cost) and is negative after costs
    np.testing.assert_allclose(
        s["net_pnl_usd"], s["gross_pnl_usd"] - s["total_cost_usd"], atol=1e-9)
    assert s["net_return_on_capital"] < 0

    # parquet exists with the documented columns
    df = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    for c in ("book_margin", "net_equity", "net_return", "daily_return"):
        assert c in df.columns
    assert len(df) > 0
    assert np.isfinite(s["capital_base_usd"]) and s["capital_base_usd"] > 0
