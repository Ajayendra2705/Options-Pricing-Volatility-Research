"""
Day 4 tests: finite-difference cross-check of ALL analytic Greeks, tight tol.

Strategy: every Greek is checked as a central first difference of an
already-verified analytic function one derivative-order below:
    delta, vega, theta, rho  <- FD of price
    gamma                    <- FD of delta  (+ looser 2nd-diff of price)
    vanna                    <- FD of delta w.r.t. sigma, and FD of vega w.r.t. F
    volga                    <- FD of vega w.r.t. sigma
First differences of analytic functions keep truncation + roundoff ~1e-9,
so rtol=1e-6 is genuinely tight.

Runs over a param grid: moneyness x maturity x vol x call/put, forward core
and spot wrappers (with carry q).
"""

import itertools

import numpy as np
import pytest

from src.greeks import black_scholes as bs

RTOL = 1e-6
ATOL = 1e-9

GRID = list(
    itertools.product(
        [80.0, 100.0, 120.0],      # F (or S)
        [0.05, 0.5, 2.0],          # T
        [0.10, 0.30, 0.80],        # sigma
        [+1, -1],                  # cp
    )
)
K, R, Q = 100.0, 0.03, 0.015


def _cfd(f, x, h):
    """Central first difference."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _step(x):
    return 1e-5 * max(1.0, abs(x))


# --- Forward core ------------------------------------------------------------

@pytest.mark.parametrize("F,T,sigma,cp", GRID)
def test_fwd_first_order_vs_price_fd(F, T, sigma, cp):
    assert bs.delta(F, K, T, sigma, R, cp) == pytest.approx(
        _cfd(lambda x: bs.price(x, K, T, sigma, R, cp), F, _step(F)), rel=RTOL, abs=ATOL
    )
    assert bs.vega(F, K, T, sigma, R) == pytest.approx(
        _cfd(lambda x: bs.price(F, K, T, x, R, cp), sigma, _step(sigma)), rel=RTOL, abs=ATOL
    )
    # theta = dV/dt = -dV/dT
    assert bs.theta(F, K, T, sigma, R, cp) == pytest.approx(
        -_cfd(lambda x: bs.price(F, K, x, sigma, R, cp), T, _step(T) * T), rel=RTOL, abs=ATOL
    )
    assert bs.rho(F, K, T, sigma, R, cp) == pytest.approx(
        _cfd(lambda x: bs.price(F, K, T, sigma, x, cp), R, _step(R)), rel=RTOL, abs=ATOL
    )


@pytest.mark.parametrize("F,T,sigma,cp", GRID)
def test_fwd_second_order_vs_first_order_fd(F, T, sigma, cp):
    # gamma = dDelta/dF
    assert bs.gamma(F, K, T, sigma, R) == pytest.approx(
        _cfd(lambda x: bs.delta(x, K, T, sigma, R, cp), F, _step(F)), rel=RTOL, abs=ATOL
    )
    # vanna = dDelta/dsigma
    assert bs.vanna(F, K, T, sigma, R) == pytest.approx(
        _cfd(lambda x: bs.delta(F, K, T, x, R, cp), sigma, _step(sigma)), rel=RTOL, abs=ATOL
    )
    # vanna symmetry: also dVega/dF (Schwarz)
    assert bs.vanna(F, K, T, sigma, R) == pytest.approx(
        _cfd(lambda x: bs.vega(x, K, T, sigma, R), F, _step(F)), rel=RTOL, abs=ATOL
    )
    # volga = dVega/dsigma
    assert bs.volga(F, K, T, sigma, R) == pytest.approx(
        _cfd(lambda x: bs.vega(F, K, T, x, R), sigma, _step(sigma)), rel=RTOL, abs=ATOL
    )


@pytest.mark.parametrize("F,T,sigma,cp", GRID)
def test_fwd_gamma_vs_price_second_diff(F, T, sigma, cp):
    # belt-and-braces: 2nd central diff of price, looser tol (roundoff ~eps/h^2)
    h = 1e-4 * F
    g2 = (
        bs.price(F + h, K, T, sigma, R, cp)
        - 2.0 * bs.price(F, K, T, sigma, R, cp)
        + bs.price(F - h, K, T, sigma, R, cp)
    ) / h**2
    assert bs.gamma(F, K, T, sigma, R) == pytest.approx(g2, rel=5e-4, abs=1e-7)


# --- Spot wrappers (with carry) -----------------------------------------------

@pytest.mark.parametrize("S,T,sigma,cp", GRID)
def test_spot_first_order_vs_price_fd(S, T, sigma, cp):
    assert bs.delta_spot(S, K, T, sigma, R, Q, cp) == pytest.approx(
        _cfd(lambda x: bs.price_spot(x, K, T, sigma, R, Q, cp), S, _step(S)), rel=RTOL, abs=ATOL
    )
    assert bs.vega_spot(S, K, T, sigma, R, Q) == pytest.approx(
        _cfd(lambda x: bs.price_spot(S, K, T, x, R, Q, cp), sigma, _step(sigma)), rel=RTOL, abs=ATOL
    )
    assert bs.theta_spot(S, K, T, sigma, R, Q, cp) == pytest.approx(
        -_cfd(lambda x: bs.price_spot(S, K, x, sigma, R, Q, cp), T, _step(T) * T), rel=RTOL, abs=ATOL
    )
    assert bs.rho_spot(S, K, T, sigma, R, Q, cp) == pytest.approx(
        _cfd(lambda x: bs.price_spot(S, K, T, sigma, x, Q, cp), R, _step(R)), rel=RTOL, abs=ATOL
    )


@pytest.mark.parametrize("S,T,sigma,cp", GRID)
def test_spot_second_order_vs_first_order_fd(S, T, sigma, cp):
    assert bs.gamma_spot(S, K, T, sigma, R, Q) == pytest.approx(
        _cfd(lambda x: bs.delta_spot(x, K, T, sigma, R, Q, cp), S, _step(S)), rel=RTOL, abs=ATOL
    )
    assert bs.vanna_spot(S, K, T, sigma, R, Q) == pytest.approx(
        _cfd(lambda x: bs.delta_spot(S, K, T, x, R, Q, cp), sigma, _step(sigma)), rel=RTOL, abs=ATOL
    )
    assert bs.volga_spot(S, K, T, sigma, R, Q) == pytest.approx(
        _cfd(lambda x: bs.vega_spot(S, K, T, x, R, Q), sigma, _step(sigma)), rel=RTOL, abs=ATOL
    )


# --- Structure / degenerate ---------------------------------------------------

def test_second_order_call_put_identical():
    F, T, sigma = 105.0, 0.7, 0.22
    # vanna/volga formulas are cp-independent; verify via FD both sides
    for greek, base in ((bs.vanna, bs.delta),):
        fd_call = _cfd(lambda x: base(F, K, T, x, R, +1), sigma, _step(sigma))
        fd_put = _cfd(lambda x: base(F, K, T, x, R, -1), sigma, _step(sigma))
        assert fd_call == pytest.approx(fd_put, rel=1e-9)
        assert greek(F, K, T, sigma, R) == pytest.approx(fd_call, rel=RTOL)


def test_zero_vol_second_order_finite():
    F = np.array([90.0, 100.0, 110.0])
    assert np.all(np.isfinite(bs.vanna(F, K, 0.5, 0.0, R)))
    assert np.all(np.isfinite(bs.volga(F, K, 0.5, 0.0, R)))
    np.testing.assert_allclose(bs.vanna(F, K, 0.5, 0.0, R), 0.0)
    np.testing.assert_allclose(bs.volga(F, K, 0.5, 0.0, R), 0.0)


def test_vanna_sign_structure():
    # OTM-forward call (F<K): d2 < 0 -> vanna > 0; ITM-forward (F>K, d2>0) -> vanna < 0
    assert bs.vanna(90.0, K, 0.5, 0.2, R) > 0
    assert bs.vanna(115.0, K, 0.5, 0.2, R) < 0


def test_volga_positive_otm_and_itm_zero_atm_dns():
    # volga = vega*d1*d2/sigma: positive away from money, ~0 where d1*d2 ~ 0
    assert bs.volga(80.0, K, 0.5, 0.2, R) > 0
    assert bs.volga(125.0, K, 0.5, 0.2, R) > 0
    sigma, T = 0.2, 0.5
    F_d1zero = K * np.exp(-0.5 * sigma**2 * T)  # d1 = 0 here
    assert abs(bs.volga(F_d1zero, K, T, sigma, R)) < 1e-10
