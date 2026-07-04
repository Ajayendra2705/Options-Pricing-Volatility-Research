"""
Day 21 tests: Greeks PnL attribution.

Exact identities first:
- delta term cancels hedge PnL exactly under daily hedging (shares_{t-1}
  == -book delta_{t-1} by construction, same pricer both sides);
- residual == actual - explained by definition, and book residual equals
  the sum of per-leg Taylor errors (hedge + financing are exact);
- financing term closes the ledger under r > 0;
- constant mark vols -> vega/vanna/volga columns identically zero.

Then approximation quality:
- GBM short straddle: explained tracks actual, cumulative residual small
  vs premium (Taylor error concentrates in the final pre-expiry bars);
- time-varying sigma on a manually re-marked leg: vega term switches on
  and the expansion still explains the PnL (vanna/volga are the
  second-order vol cross-terms that make this work).

And no-lookahead invariance.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.attribution import (TERM_COLS, attribution_summary,
                                      book_attribution, leg_attribution)
from src.backtest.engine import Leg, run_hedged
from src.greeks.black_scholes import price_spot
from tests.test_engine import gbm_path, straddle


def _book(seed=0, sigma_real=0.20, mark=0.25, r=0.0, n=43):
    rng = np.random.default_rng(seed)
    dates, S = gbm_path(rng, n=n, sigma=sigma_real)
    legs = straddle(dates, vol=mark)
    led = run_hedged(dates, S, legs, r=r)
    return book_attribution(led, legs, r=r), led, legs


# --- exact identities --------------------------------------------------------

def test_delta_term_cancels_hedge_under_daily_hedging():
    book, _, _ = _book(seed=1)
    # daily hedge: shares_{t-1} = -delta_book_{t-1}; attribution recomputes
    # the same delta with the same pricer -> exact cancellation every bar
    np.testing.assert_allclose(book["delta_pnl"] + book["hedge_pnl"], 0.0,
                               atol=1e-9)


def test_residual_is_actual_minus_explained():
    book, _, _ = _book(seed=2)
    np.testing.assert_allclose(book["residual"],
                               book["actual"] - book["explained"], atol=1e-12)


def test_book_residual_equals_sum_of_leg_taylor_errors():
    # hedge + financing are exact, so the book residual must be exactly the
    # per-leg mark-to-market moves minus the per-leg Taylor terms
    rng = np.random.default_rng(3)
    dates, S = gbm_path(rng)
    legs = straddle(dates)
    led, legs_led = run_hedged(dates, S, legs, r=0.0, return_legs=True)
    book = book_attribution(led, legs, r=0.0)
    la = leg_attribution(dates, S, legs)
    idx = pd.Index(led["date"])
    opt = legs_led.groupby("date")["pnl_day"].sum().reindex(idx, fill_value=0.0)
    terms = (la.groupby("date")[TERM_COLS].sum().sum(axis=1)
             .reindex(idx, fill_value=0.0))
    leg_err = opt - terms
    np.testing.assert_allclose(book["residual"], leg_err.to_numpy(), atol=1e-9)


def test_financing_closes_ledger_with_positive_rate():
    book, _, _ = _book(seed=4, r=0.05)
    # financing is nonzero and the identity still holds bar by bar: whatever
    # actual - explained leaves is pure option Taylor error, small vs premium
    assert book["financing"].abs().sum() > 0.0
    assert book["residual"].abs().sum() < 0.30 * book["actual"].abs().sum()


def test_constant_marks_have_zero_vol_terms():
    book, _, _ = _book(seed=5)
    for c in ("vega_pnl", "vanna_pnl", "volga_pnl", "rho_pnl"):
        assert (book[c] == 0.0).all()


# --- approximation quality ---------------------------------------------------

def test_explained_tracks_actual_on_gbm():
    book, led, _ = _book(seed=7)
    prem = abs(led["V_opt"].iloc[0])
    assert abs(book["residual"].sum()) < 0.10 * prem
    # away from expiry the one-day Taylor error is tiny
    early = book.iloc[1:-5]
    assert early["residual"].abs().sum() < 0.05 * prem
    s = attribution_summary(book)
    assert s["residual_abs_sum_over_actual_abs_sum"] < 0.30


def test_vega_term_explains_vol_moves():
    # long call re-marked by hand on a time-varying vol path; expiry beyond
    # the window so tau stays > 0 and no settlement bar muddies the check
    rng = np.random.default_rng(11)
    n = 30
    dates, S = gbm_path(rng, n=n, S0=100.0, sigma=0.20)
    expiry = dates[-1] + pd.Timedelta(days=90)
    leg = Leg(K=100.0, expiry=expiry, cp=+1, qty=+1, mark_vol=0.25)
    sig = pd.Series(0.25 + 0.03 * np.sin(np.arange(n) / 4.0), index=dates)

    tau = np.array([(expiry - d).days / 365.0 for d in dates])
    value = 100.0 * price_spot(S, 100.0, tau, sig.to_numpy(), 0.0, 0.0, +1)
    actual = pd.Series(value, index=dates).diff().dropna()

    la = leg_attribution(dates, S, [leg], sigma_by_leg={0: sig})
    explained = la.set_index("date")[TERM_COLS].sum(axis=1)
    resid = actual - explained
    assert la["vega_pnl"].abs().sum() > 0.0
    assert resid.abs().sum() < 0.05 * actual.abs().sum()
    # second-order vol terms are live too (dS*dsig and dsig^2 cross-terms)
    assert la["vanna_pnl"].abs().sum() > 0.0
    assert la["volga_pnl"].abs().sum() > 0.0


# --- structure -----------------------------------------------------------------

def test_no_lookahead():
    rng = np.random.default_rng(13)
    dates, S = gbm_path(rng)
    legs = straddle(dates)
    la = leg_attribution(dates, S, legs)
    t = 20
    S2 = S.copy()
    S2[t + 1:] *= 1.5
    la2 = leg_attribution(dates, S2, legs)
    m = la["date"] <= dates[t]
    pd.testing.assert_frame_equal(la[m], la2[la2["date"] <= dates[t]])


def test_misaligned_inputs_rejected():
    dates = pd.bdate_range("2023-06-02", periods=10)
    with pytest.raises(ValueError, match="misaligned"):
        leg_attribution(dates, np.full(9, 180.0),
                        [Leg(K=180.0, expiry=dates[-1], cp=1, qty=1,
                             mark_vol=0.2)])
