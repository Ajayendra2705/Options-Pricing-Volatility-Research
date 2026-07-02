"""
Day 3 tests: Black-Scholes pricing + first-order Greeks.

- Textbook values (Hull, "Options, Futures and Other Derivatives"):
  * Prices: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81
  * Greeks: S=49, K=50, r=5%, sigma=20%, T=0.3846 -> delta .522, gamma .066,
    vega 12.1, theta -4.31, rho 8.91 (call)
- Put-call parity: C - P = df * (F - K), grid of params.
- Forward-core vs spot wrapper consistency.
- Degenerate limits: sigma->0 gives discounted intrinsic.
"""

import numpy as np
import pytest

from src.greeks import black_scholes as bs


# --- Textbook prices (Hull ch. 15 example) ---------------------------------

HULL_PRICE = dict(S=42.0, K=40.0, T=0.5, sigma=0.20, r=0.10, q=0.0)


def test_hull_call_price():
    c = bs.price_spot(cp=+1, **HULL_PRICE)
    assert c == pytest.approx(4.76, abs=0.005)


def test_hull_put_price():
    p = bs.price_spot(cp=-1, **HULL_PRICE)
    assert p == pytest.approx(0.81, abs=0.005)


# --- Textbook Greeks (Hull ch. 19 running example) --------------------------

HULL_GREEKS = dict(S=49.0, K=50.0, T=0.3846, sigma=0.20, r=0.05, q=0.0)


def test_hull_call_delta():
    assert bs.delta_spot(cp=+1, **HULL_GREEKS) == pytest.approx(0.522, abs=0.001)


def test_hull_gamma():
    assert bs.gamma_spot(**HULL_GREEKS) == pytest.approx(0.066, abs=0.001)


def test_hull_vega():
    # Hull quotes vega per 1.0 of vol: 12.1
    assert bs.vega_spot(**HULL_GREEKS) == pytest.approx(12.1, abs=0.05)


def test_hull_call_theta():
    # Hull quotes theta per year: -4.31
    assert bs.theta_spot(cp=+1, **HULL_GREEKS) == pytest.approx(-4.31, abs=0.01)


def test_hull_call_rho():
    assert bs.rho_spot(cp=+1, **HULL_GREEKS) == pytest.approx(8.91, abs=0.01)


def test_put_delta_relation():
    # delta_put = delta_call - exp(-qT)
    d_c = bs.delta_spot(cp=+1, **HULL_GREEKS)
    d_p = bs.delta_spot(cp=-1, **HULL_GREEKS)
    q, T = HULL_GREEKS["q"], HULL_GREEKS["T"]
    assert d_p == pytest.approx(d_c - np.exp(-q * T), abs=1e-12)


# --- Put-call parity over a grid --------------------------------------------

def test_put_call_parity_forward_grid():
    F = np.array([80.0, 100.0, 125.0])[:, None, None]
    K = np.array([70.0, 100.0, 140.0])[None, :, None]
    sigma = np.array([0.05, 0.2, 0.8])[None, None, :]
    T, r = 0.75, 0.03
    c = bs.price(F, K, T, sigma, r, +1)
    p = bs.price(F, K, T, sigma, r, -1)
    parity = np.exp(-r * T) * (F - K) * np.ones_like(sigma)
    np.testing.assert_allclose(c - p, parity, atol=1e-10)


def test_put_call_parity_spot_with_carry():
    S, K, T, sigma, r, q = 100.0, 95.0, 1.25, 0.3, 0.04, 0.015
    c = bs.price_spot(S, K, T, sigma, r, q, +1)
    p = bs.price_spot(S, K, T, sigma, r, q, -1)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


# --- Forward core vs spot wrapper -------------------------------------------

def test_spot_wrapper_matches_forward_core():
    S, K, T, sigma, r, q = 100.0, 110.0, 0.5, 0.25, 0.05, 0.02
    F = bs.forward(S, r, q, T)
    assert F == pytest.approx(S * np.exp((r - q) * T))
    for cp in (+1, -1):
        assert bs.price_spot(S, K, T, sigma, r, q, cp) == pytest.approx(
            bs.price(F, K, T, sigma, r, cp), abs=1e-12
        )
    # spot delta = forward delta * dF/dS = delta_fwd * exp((r-q)T)
    assert bs.delta_spot(S, K, T, sigma, r, q, +1) == pytest.approx(
        bs.delta(F, K, T, sigma, r, +1) * np.exp((r - q) * T), abs=1e-12
    )
    # spot gamma = forward gamma * (dF/dS)^2
    assert bs.gamma_spot(S, K, T, sigma, r, q) == pytest.approx(
        bs.gamma(F, K, T, sigma, r) * np.exp(2 * (r - q) * T), abs=1e-12
    )
    # vega identical
    assert bs.vega_spot(S, K, T, sigma, r, q) == pytest.approx(
        bs.vega(F, K, T, sigma, r), abs=1e-12
    )


# --- Degenerate limits -------------------------------------------------------

def test_zero_vol_collapses_to_discounted_intrinsic():
    T, r = 0.5, 0.05
    df = np.exp(-r * T)
    F = np.array([90.0, 100.0, 110.0])
    K = 100.0
    c = bs.price(F, K, T, 0.0, r, +1)
    p = bs.price(F, K, T, 0.0, r, -1)
    np.testing.assert_allclose(c, df * np.maximum(F - K, 0.0), atol=1e-12)
    np.testing.assert_allclose(p, df * np.maximum(K - F, 0.0), atol=1e-12)
    assert np.all(np.isfinite(bs.gamma(F, K, T, 0.0, r)))
    assert np.all(np.isfinite(bs.theta(F, K, T, 0.0, r, +1)))


def test_deep_itm_call_delta_near_df():
    # F >> K: forward call delta -> exp(-rT)
    T, r = 1.0, 0.05
    d = bs.delta(1000.0, 10.0, T, 0.2, r, +1)
    assert d == pytest.approx(np.exp(-r * T), abs=1e-9)
