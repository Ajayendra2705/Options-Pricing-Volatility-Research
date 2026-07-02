"""
Day 5 tests: price -> IV -> price round-trip, err < 1e-6 (PLAN.md gate).

Covers: moneyness x maturity x vol grid (forward core), spot wrapper with
carry, no-arb bound rejection (nan), intrinsic edge (0.0), deep wings.
"""

import itertools

import numpy as np
import pytest

from src.greeks import black_scholes as bs
from src.greeks.iv_invert import implied_vol, implied_vol_spot

R, Q = 0.04, 0.015

GRID = list(
    itertools.product(
        [60.0, 85.0, 100.0, 115.0, 160.0],   # F
        [100.0],                              # K
        [0.02, 0.25, 1.0, 3.0],               # T
        [0.05, 0.20, 0.60, 1.50],             # sigma_true
        [+1, -1],                             # cp
    )
)


@pytest.mark.parametrize("F,K,T,sigma,cp", GRID)
def test_roundtrip_price_error_below_1e6(F, K, T, sigma, cp):
    p = bs.price(F, K, T, sigma, R, cp)
    iv = implied_vol(p, F, K, T, R, cp)
    df = np.exp(-R * T)
    intrinsic = df * max(cp * (F - K), 0.0)
    tv = p - intrinsic                   # time value: all the sigma information
    if tv < 1e-12:
        # premium numerically at intrinsic (ultra-deep wing): sigma unidentifiable
        assert iv == 0.0 or np.isnan(iv) or abs(bs.price(F, K, T, iv, R, cp) - p) < 1e-6
        return
    assert np.isfinite(iv)
    p_back = bs.price(F, K, T, iv, R, cp)
    assert abs(p_back - p) < 1e-6        # PLAN.md gate: always
    if tv > 1e-8 * max(1.0, p):
        # time value well above float noise of the premium -> sigma identifiable
        assert iv == pytest.approx(sigma, rel=1e-6, abs=1e-8)
    else:
        # deep ITM: tv sits at ulp(p), sigma only recoverable to ~ulp(p)/vega
        assert iv == pytest.approx(sigma, rel=1e-4)


def test_roundtrip_vectorized():
    rng_F = np.array([80.0, 100.0, 120.0])
    sigma = np.array([0.15, 0.30, 0.45])
    T, cp = 0.5, +1
    p = bs.price(rng_F, 100.0, T, sigma, R, cp)
    iv = implied_vol(p, rng_F, 100.0, T, R, cp)
    np.testing.assert_allclose(iv, sigma, rtol=1e-8)


def test_spot_wrapper_roundtrip_with_carry():
    S, K, T, sigma = 100.0, 92.0, 0.75, 0.28
    for cp in (+1, -1):
        p = bs.price_spot(S, K, T, sigma, R, Q, cp)
        iv = implied_vol_spot(p, S, K, T, R, Q, cp)
        assert iv == pytest.approx(sigma, rel=1e-8)
        assert abs(bs.price_spot(S, K, T, iv, R, Q, cp) - p) < 1e-6


# --- No-arb bounds / edges ----------------------------------------------------

def test_price_below_intrinsic_is_nan():
    F, K, T = 120.0, 100.0, 0.5
    df = np.exp(-R * T)
    below = df * (F - K) * 0.98          # call premium below discounted intrinsic
    assert np.isnan(implied_vol(below, F, K, T, R, +1))


def test_price_above_upper_bound_is_nan():
    F, K, T = 100.0, 100.0, 0.5
    df = np.exp(-R * T)
    assert np.isnan(implied_vol(df * F * 1.01, F, K, T, R, +1))   # call > df*F
    assert np.isnan(implied_vol(df * K * 1.01, F, K, T, R, -1))   # put > df*K


def test_price_at_intrinsic_gives_zero_vol():
    F, K, T = 120.0, 100.0, 0.5
    df = np.exp(-R * T)
    assert implied_vol(df * (F - K), F, K, T, R, +1) == 0.0


def test_invalid_inputs_are_nan():
    assert np.isnan(implied_vol(np.nan, 100.0, 100.0, 0.5, R, +1))
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, 0.0, R, +1))   # T = 0


def test_deep_wing_tiny_premium():
    # far-OTM put, tiny but resolvable premium
    F, K, T, sigma = 100.0, 55.0, 0.25, 0.35
    p = bs.price(F, K, T, sigma, R, -1)
    assert p < 2e-3
    iv = implied_vol(p, F, K, T, R, -1)
    assert iv == pytest.approx(sigma, rel=1e-6)
