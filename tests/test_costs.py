"""
Day 24 — Transaction-cost tests.

1. Half-spread formula: 0.5*(ask-bid)*mult*|qty| summed over call+put.
2. Commission: 2 * per_contract * |qty| on entry.
3. Net = gross - total cost, exactly.
4. Costs are always a non-negative drag (crossing spread costs both sides).
5. Hedge slippage: zero at primary bps=0, positive and correctly scaled when on.
6. Real-data gate (skip if files missing): 10 positions, net < gross, costs > 0,
   net PnL turns negative (the disproof punchline).
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import (
    _load_cost_params,
    hedge_slippage_cost,
    option_entry_cost,
)
from src.backtest.reconcile import PROCESSED_DIR, RAW_DIR


# ── synthetic helpers ────────────────────────────────────────────────────────

def _chain_row(date, expiry, K, cp, bid, ask):
    return {"date": date, "expiry": expiry, "strike": K,
            "option_type": cp, "bid": bid, "ask": ask}


def _make_chain(date, expiry, K, c_bid, c_ask, p_bid, p_ask):
    return pd.DataFrame([
        _chain_row(date, expiry, K, "C", c_bid, c_ask),
        _chain_row(date, expiry, K, "P", p_bid, p_ask),
    ])


def _make_pos(K=180.0, qty=-1.0):
    return {"date": pd.Timestamp("2023-06-02"),
            "expiry": pd.Timestamp("2023-06-16"),
            "K": K, "qty": qty, "side": "short_vol" if qty < 0 else "long_vol"}


# ── 1. half-spread ───────────────────────────────────────────────────────────

def test_half_spread_formula():
    pos = _make_pos(qty=-1.0)
    chain = _make_chain(pos["date"], pos["expiry"], pos["K"],
                        c_bid=5.0, c_ask=5.4, p_bid=4.0, p_ask=4.2)
    ec = option_entry_cost(pos, chain, {"commission_per_contract_usd": 0.65})
    # half spreads: call 0.2, put 0.1 -> (0.2+0.1)*100*1 = 30
    np.testing.assert_allclose(ec["half_spread"], 30.0, rtol=1e-12)


def test_half_spread_scales_with_qty():
    chain = _make_chain(pd.Timestamp("2023-06-02"), pd.Timestamp("2023-06-16"),
                        180.0, 5.0, 5.4, 4.0, 4.2)
    p1 = _make_pos(qty=-1.0)
    p3 = _make_pos(qty=-3.0)
    e1 = option_entry_cost(p1, chain, {"commission_per_contract_usd": 0.65})
    e3 = option_entry_cost(p3, chain, {"commission_per_contract_usd": 0.65})
    np.testing.assert_allclose(e3["half_spread"], 3.0 * e1["half_spread"])


# ── 2. commission ────────────────────────────────────────────────────────────

def test_commission():
    pos = _make_pos(qty=-2.0)
    chain = _make_chain(pos["date"], pos["expiry"], pos["K"],
                        5.0, 5.4, 4.0, 4.2)
    ec = option_entry_cost(pos, chain, {"commission_per_contract_usd": 0.65})
    # 2 legs * 0.65 * |qty|=2 -> 2.60
    np.testing.assert_allclose(ec["commission"], 2.60, rtol=1e-12)


# ── 3. costs are a non-negative drag ─────────────────────────────────────────

def test_cost_non_negative_both_sides():
    chain = _make_chain(pd.Timestamp("2023-06-02"), pd.Timestamp("2023-06-16"),
                        180.0, 5.0, 5.4, 4.0, 4.2)
    for qty in (-1.0, +1.0):        # short and long both pay to cross
        ec = option_entry_cost(_make_pos(qty=qty), chain,
                               {"commission_per_contract_usd": 0.65})
        assert ec["total"] > 0
        assert ec["half_spread"] >= 0
        assert ec["commission"] >= 0


# ── 5. hedge slippage ────────────────────────────────────────────────────────

def test_hedge_slippage_zero_in_primary():
    led = pd.DataFrame({"traded": [10.0, -3.0, 5.0], "S": [180.0, 181.0, 179.0]})
    assert hedge_slippage_cost(led, {"underlying_slippage_bps": 0.0}) == 0.0


def test_hedge_slippage_scaled():
    led = pd.DataFrame({"traded": [10.0, -3.0], "S": [180.0, 200.0]})
    # notional = 10*180 + 3*200 = 1800 + 600 = 2400; 5 bps -> 2400*5e-4 = 1.2
    cost = hedge_slippage_cost(led, {"underlying_slippage_bps": 5.0})
    np.testing.assert_allclose(cost, 1.2, rtol=1e-12)


# ── config parse ─────────────────────────────────────────────────────────────

def test_cost_params_from_config():
    params = _load_cost_params()
    # pre-registered commission is 0.65; primary underlying slippage is 0
    assert params["commission_per_contract_usd"] == pytest.approx(0.65)
    assert params["underlying_slippage_bps"] == 0.0


# ── 6. real-data gate ────────────────────────────────────────────────────────

_NEEDED = [PROCESSED_DIR / "signal.parquet",
           PROCESSED_DIR / "forwards.parquet",
           PROCESSED_DIR / "svi_params_joint.parquet",
           PROCESSED_DIR / "chain_clean.parquet",
           RAW_DIR / "aapl_ohlc.parquet",
           RAW_DIR / "aapl_ohlc_ext.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="real data files not present")
def test_real_data_costs():
    from src.backtest.costs import run_costs

    summary = run_costs()

    assert summary["n_positions"] == 10
    # costs strictly reduce PnL
    assert summary["total_cost"] > 0
    np.testing.assert_allclose(
        summary["net_pnl"], summary["gross_pnl"] - summary["total_cost"],
        rtol=0, atol=1e-9)
    # gross was ~breakeven (+$3.87); after $400+ of costs, net is clearly negative
    assert summary["net_pnl"] < 0
    assert summary["net_pnl"] < summary["gross_pnl"]
    # primary: no hedge slippage (zero underlying spread)
    assert summary["total_hedge_slippage"] == 0.0
    # per-position identity
    for r in summary["positions"]:
        np.testing.assert_allclose(
            r["net_pnl"], r["gross_pnl"] - r["total_cost"], atol=1e-9)
