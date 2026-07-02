"""
Day 11 tests: Durrleman no-butterfly detection (PLAN.md: known-bad params flagged).

- Known-bad: Axel Vogt params (Gatheral-Jacquier arbitrage example) -> flagged.
- Known-good: flat smile, benign SVI, all Day-10 real fits structurally valid runs.
- Formula validation:
  * w', w'' analytic vs finite difference
  * g(k) sign vs Breeden-Litzenberger density (FD second derivative of
    Black call prices in strike) — fully independent arbitrage detector.
"""

import numpy as np
import pytest

from src.greeks import black_scholes as bs
from src.surface.no_arb import (
    check_butterfly,
    durrleman_g,
    svi_w_prime,
    svi_w_second,
)
from src.surface.svi import SVIParams, svi_total_variance

# Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces":
# Axel Vogt's example of raw-SVI butterfly arbitrage.
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
BENIGN = SVIParams(a=0.015, b=0.35, rho=-0.65, m=0.05, sigma=0.25)


def test_vogt_params_flagged():
    rep = check_butterfly(VOGT)
    assert not rep["arb_free"]
    assert rep["min_g"] < 0
    assert rep["n_violations"] > 0


def test_flat_smile_arb_free():
    # w constant: g = 1 everywhere
    flat = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    k = np.linspace(-1, 1, 51)
    np.testing.assert_allclose(durrleman_g(k, flat), 1.0, atol=1e-12)
    assert check_butterfly(flat)["arb_free"]


def test_benign_params_arb_free():
    rep = check_butterfly(BENIGN)
    assert rep["arb_free"], rep


def test_negative_w_reported_not_masked():
    # a very negative: w < 0 somewhere -> g = -inf there, flagged
    bad_w = SVIParams(a=-0.5, b=0.1, rho=0.0, m=0.0, sigma=0.1)
    rep = check_butterfly(bad_w)
    assert not rep["arb_free"]
    assert rep["min_w"] < 0
    assert rep["min_g"] == -np.inf


# --- derivative formulas vs finite difference --------------------------------

@pytest.mark.parametrize("params", [VOGT, BENIGN])
def test_w_derivatives_match_fd(params):
    k = np.linspace(-0.8, 0.8, 17)
    h = 1e-6
    wp_fd = (svi_total_variance(k + h, params) - svi_total_variance(k - h, params)) / (2 * h)
    np.testing.assert_allclose(svi_w_prime(k, params), wp_fd, rtol=1e-7, atol=1e-10)
    wpp_fd = (svi_w_prime(k + h, params) - svi_w_prime(k - h, params)) / (2 * h)
    np.testing.assert_allclose(svi_w_second(k, params), wpp_fd, rtol=1e-6, atol=1e-10)


# --- independent detector: Breeden-Litzenberger density ----------------------

@pytest.mark.parametrize("params,expect_arb", [(VOGT, True), (BENIGN, False)])
def test_g_sign_matches_bl_density(params, expect_arb):
    """Butterfly arb iff risk-neutral density (d2C/dK2) goes negative.
    Compute undiscounted Black calls on F=1 from the SVI vols (T=1 so
    w = sigma^2), FD the density, compare against g's verdict."""
    T = 1.0
    k = np.linspace(-1.2, 1.2, 2401)
    K = np.exp(k)
    iv = np.sqrt(np.maximum(svi_total_variance(k, params), 1e-12) / T)
    C = bs.price(1.0, K, T, iv, 0.0, 1)
    dK = np.diff(K)
    dens = 2.0 * np.diff(np.diff(C) / dK) / (dK[1:] + dK[:-1])   # nonuniform 2nd diff
    has_neg_density = dens.min() < -1e-8
    assert has_neg_density == expect_arb
    # and Durrleman agrees on the same interior grid
    g = durrleman_g(k[1:-1], params)
    assert (g.min() < 0) == expect_arb
    # sign agreement where density is decisively signed
    decisive = np.abs(dens) > 1e-6
    sign_match = np.mean(np.sign(dens[decisive]) == np.sign(g[decisive]))
    assert sign_match > 0.99
