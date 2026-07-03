"""
Day 19 tests: delta-hedge engine.

Exact identities first (these catch accounting bugs cold):
- synthetic forward (long call + short put) is delta-hedged perfectly ->
  equity identically zero along the whole path;
- flat price path -> short straddle keeps exactly the entry premium;
- ledger algebra: equity column == cash + shares*S + V_opt, cash deltas ==
  -traded*S (+ settlement), book ends settled (shares 0, V 0).

Then the statistical physics of delta hedging on GBM paths:
- realized == implied -> mean hedged PnL ~ 0;
- realized below/above implied -> short vol wins/loses;
- hedging less often inflates PnL dispersion ~ sqrt(interval).

And no-lookahead invariance (future prices corrupted -> past rows identical).
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Leg, run_hedged
from src.greeks.black_scholes import price_spot

MULT = 100.0


def gbm_path(rng, n=43, S0=180.0, sigma=0.20, start="2023-06-02"):
    dates = pd.bdate_range(start, periods=n)
    dt = 1.0 / 252
    incr = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(n - 1)
    return dates, S0 * np.exp(np.r_[0.0, np.cumsum(incr)])


def straddle(dates, K=180.0, vol=0.25, qty=-1):
    return [Leg(K=K, expiry=dates[-1], cp=+1, qty=qty, mark_vol=vol),
            Leg(K=K, expiry=dates[-1], cp=-1, qty=qty, mark_vol=vol)]


def _pnls(n_paths, sigma_real, sigma_mark, hedge_every=1, seed=3):
    rng = np.random.default_rng(seed)
    out = np.empty(n_paths)
    for i in range(n_paths):
        dates, S = gbm_path(rng, sigma=sigma_real)
        led = run_hedged(dates, S, straddle(dates, vol=sigma_mark),
                         hedge_every=hedge_every)
        out[i] = led["equity"].iloc[-1]
    return out


# --- exact identities ---------------------------------------------------------

def test_synthetic_forward_hedges_to_zero():
    rng = np.random.default_rng(0)
    dates, S = gbm_path(rng, sigma=0.35)
    legs = [Leg(K=175.0, expiry=dates[-1], cp=+1, qty=+1, mark_vol=0.25),
            Leg(K=175.0, expiry=dates[-1], cp=-1, qty=-1, mark_vol=0.25)]
    led = run_hedged(dates, S, legs, r=0.0)
    # C - P = S - K (r=0): delta exactly 1, the hedge replicates it exactly,
    # equity must be zero to machine precision on EVERY bar
    np.testing.assert_allclose(led["equity"], 0.0, atol=1e-9)


def test_flat_path_keeps_premium():
    dates = pd.bdate_range("2023-06-02", periods=30)
    S = np.full(30, 180.0)
    legs = straddle(dates, K=180.0, vol=0.25, qty=-1)
    led = run_hedged(dates, S, legs, r=0.0)
    T0 = (dates[-1] - dates[0]).days / 365.0
    prem = MULT * sum(price_spot(180.0, 180.0, T0, 0.25, 0.0, 0.0, cp)
                      for cp in (+1, -1))
    # options expire worthless at the strike; constant price -> hedge nets 0
    assert led["equity"].iloc[-1] == pytest.approx(prem, abs=1e-9)
    assert led["shares"].iloc[-1] == 0.0 and led["V_opt"].iloc[-1] == 0.0


def test_ledger_algebra():
    rng = np.random.default_rng(1)
    dates, S = gbm_path(rng)
    led = run_hedged(dates, S, straddle(dates), r=0.0)
    np.testing.assert_allclose(
        led["equity"], led["cash"] + led["shares"] * led["S"] + led["V_opt"],
        atol=1e-12)
    # r=0: cash moves only via trades, entry premium, settlement
    dcash = led["cash"].diff().iloc[1:-1]
    np.testing.assert_allclose(
        dcash, (-led["traded"] * led["S"]).iloc[1:-1], atol=1e-9)
    # end state fully settled
    assert led["shares"].iloc[-1] == 0.0 and led["V_opt"].iloc[-1] == 0.0
    assert led["equity"].iloc[-1] == pytest.approx(led["cash"].iloc[-1])


# --- hedging physics on GBM ----------------------------------------------------

N_MC = 150


@pytest.fixture(scope="module")
def pnl_breakeven():
    return _pnls(N_MC, sigma_real=0.25, sigma_mark=0.25)


def test_breakeven_when_realized_equals_implied(pnl_breakeven):
    pnl = pnl_breakeven
    se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    assert abs(pnl.mean()) < 3 * se


def test_short_vol_wins_when_realized_below_implied():
    pnl = _pnls(N_MC, sigma_real=0.15, sigma_mark=0.30)
    se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    assert pnl.mean() > 3 * se


def test_short_vol_loses_when_realized_above_implied():
    pnl = _pnls(N_MC, sigma_real=0.45, sigma_mark=0.30)
    se = pnl.std(ddof=1) / np.sqrt(len(pnl))
    assert pnl.mean() < -3 * se


def test_coarser_hedging_inflates_dispersion(pnl_breakeven):
    weekly = _pnls(N_MC, 0.25, 0.25, hedge_every=5)
    # Leland: hedge-error std ~ sqrt(interval); sqrt(5) ~ 2.24
    assert 1.4 < weekly.std(ddof=1) / pnl_breakeven.std(ddof=1) < 3.5


# --- structure -----------------------------------------------------------------

def test_no_lookahead():
    rng = np.random.default_rng(2)
    dates, S = gbm_path(rng)
    led = run_hedged(dates, S, straddle(dates))
    t = 20
    S2 = S.copy()
    S2[t + 1:] *= 1.5
    led2 = run_hedged(dates, S2, straddle(dates))
    pd.testing.assert_frame_equal(led.iloc[: t + 1], led2.iloc[: t + 1])


def test_validation():
    dates = pd.bdate_range("2023-06-02", periods=10)
    S = np.full(10, 180.0)
    legs = [Leg(K=180, expiry=dates[-1] + pd.Timedelta(days=30), cp=1,
                qty=1, mark_vol=0.2)]
    with pytest.raises(ValueError, match="last expiry"):
        run_hedged(dates, S, legs)
    legs = [Leg(K=180, expiry=dates[-1], cp=1, qty=1, mark_vol=0.2)]
    with pytest.raises(ValueError, match="misaligned"):
        run_hedged(dates, S[:-1], legs)
