"""
Day 20 tests: per-leg ledger, gamma-weighted RV, break-even vol.

- Per-leg marks sum exactly to the book (value, delta); per-leg daily PnL
  plus hedge-holding PnL reproduces the equity path bar by bar (r=0).
- Gamma-weighted RV recovers sigma on GBM (its weights are valid weights).
- Theta-gamma first-order PnL tracks engine equity (slope ~ 1, high corr)
  and the cumulative gap is small relative to premium.
- Break-even consistency across paths: short-gamma book profits exactly
  when sigma_gw < sigma_mark (first order -> high agreement rate, not 100%).
- Flat path: sigma_gw = 0, per-leg PnL = each leg's premium.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import Leg, run_hedged
from src.backtest.pnl import breakeven_report, gamma_weighted_rv, theta_gamma_pnl
from tests.test_engine import gbm_path, straddle


def _run(seed=0, sigma_real=0.20, mark=0.25, n=43):
    rng = np.random.default_rng(seed)
    dates, S = gbm_path(rng, n=n, sigma=sigma_real)
    legs = straddle(dates, vol=mark)
    led, legs_led = run_hedged(dates, S, legs, r=0.0, return_legs=True)
    return led, legs_led, legs


def test_legs_sum_to_book():
    led, legs_led, _ = _run()
    by_date = legs_led.groupby("date")[["value", "delta"]].sum()
    np.testing.assert_allclose(by_date["value"], led.set_index("date")["V_opt"],
                               atol=1e-9)
    np.testing.assert_allclose(by_date["delta"], led.set_index("date")["delta_opt"],
                               atol=1e-9)


def test_per_leg_pnl_plus_hedge_reproduces_equity():
    led, legs_led, _ = _run()
    opt_pnl = legs_led.groupby("date")["pnl_day"].sum()
    led = led.set_index("date")
    hedge_pnl = (led["shares"].shift(1) * led["S"].diff()).fillna(0.0)
    recon = (opt_pnl.reindex(led.index, fill_value=0.0) + hedge_pnl).cumsum()
    np.testing.assert_allclose(recon, led["equity"], atol=1e-9)


def test_gamma_weighted_rv_recovers_sigma():
    # average across paths; weights valid -> estimator centred on sigma_real.
    # trading-vs-calendar clock mismatch of the test GBM (42/252 trading yrs
    # over 59/365 calendar yrs) biases it ~ +1.5% by construction; allow 5%.
    vals = [gamma_weighted_rv(*_run(seed=s, sigma_real=0.30)[:2])
            for s in range(40)]
    assert np.mean(vals) == pytest.approx(0.30, rel=0.05)


def test_theta_gamma_tracks_equity():
    led, legs_led, legs = _run(seed=7)
    tg = theta_gamma_pnl(led, legs_led, {j: l.mark_vol for j, l in enumerate(legs)})
    dpnl = led.set_index("date")["equity"].diff().dropna()
    tg = tg.reindex(dpnl.index)
    corr = np.corrcoef(tg, dpnl)[0, 1]
    slope = np.polyfit(tg, dpnl, 1)[0]
    assert corr > 0.95
    assert 0.8 < slope < 1.2
    prem = abs(led["V_opt"].iloc[0])
    assert abs(tg.sum() - led["equity"].iloc[-1]) < 0.15 * prem


def test_breakeven_consistency_across_paths():
    agree = []
    for s in range(60):
        led, legs_led, legs = _run(seed=s, sigma_real=0.25, mark=0.25)
        rep = breakeven_report(led, legs_led,
                               {j: l.mark_vol for j, l in enumerate(legs)})
        agree.append(rep["pnl_consistent_with_gap"])
        assert rep["net_gamma_sign"] == -1.0            # short straddle
    # first-order identity -> not 100%, but strongly dominant
    assert np.mean(agree) > 0.85


def test_flat_path_per_leg():
    dates = pd.bdate_range("2023-06-02", periods=30)
    S = np.full(30, 180.0)
    legs = straddle(dates, K=180.0, vol=0.25, qty=-1)
    led, legs_led = run_hedged(dates, S, legs, r=0.0, return_legs=True)
    rep = breakeven_report(led, legs_led, {0: 0.25, 1: 0.25})
    assert rep["sigma_gamma_weighted"] == 0.0
    # each short leg keeps exactly its own premium: marks run v0 -> 0, so
    # summed daily PnL = -v0 (positive for a short leg)
    for j in (0, 1):
        leg_j = legs_led[legs_led["leg"] == j]
        assert leg_j["pnl_day"].sum() == pytest.approx(
            -leg_j["value"].iloc[0], abs=1e-9)


def test_duplicate_leg_object_rejected():
    dates = pd.bdate_range("2023-06-02", periods=10)
    S = np.full(10, 180.0)
    leg = Leg(K=180.0, expiry=dates[-1], cp=1, qty=-1, mark_vol=0.2)
    with pytest.raises(ValueError, match="distinct Leg"):
        run_hedged(dates, S, [leg, leg])
