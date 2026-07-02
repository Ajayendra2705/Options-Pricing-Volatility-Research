"""
Day 5 tests: synthetic known-sigma recovery (PLAN.md deliverable).

Generate a synthetic option chain with a known smile sigma(K), price it
exactly under Black-76, invert every quote, assert the smile is recovered.
Fixed seed for the randomized sweep.
"""

import numpy as np
import pytest

from src.greeks import black_scholes as bs
from src.greeks.iv_invert import implied_vol
from src.utils.seed import set_global_seed

R = 0.03


def smile(K, F=100.0, atm=0.20, skew=-0.002, curv=8e-5):
    """Deterministic quadratic smile in strike, vol floor 5%."""
    k = K - F
    return np.maximum(atm + skew * k + curv * k**2, 0.05)


def test_synthetic_chain_smile_recovery():
    F, T = 100.0, 0.5
    K = np.arange(60.0, 145.0, 5.0)
    sigma_true = smile(K)
    for cp in (+1, -1):
        p = bs.price(F, K, T, sigma_true, R, cp)
        iv = implied_vol(p, F, K, T, R, cp)
        np.testing.assert_allclose(iv, sigma_true, rtol=1e-7, atol=1e-9)


def test_randomized_recovery_sweep():
    set_global_seed()   # reproducible
    n = 500
    F = np.random.uniform(50.0, 200.0, n)
    K = F * np.random.uniform(0.6, 1.5, n)
    T = np.random.uniform(0.02, 3.0, n)
    sigma = np.random.uniform(0.05, 1.2, n)
    r = np.random.uniform(0.0, 0.08, n)
    cp = np.where(np.random.rand(n) < 0.5, 1, -1)
    p = bs.price(F, K, T, sigma, r, cp)
    iv = implied_vol(p, F, K, T, r, cp)

    df = np.exp(-r * T)
    intrinsic = df * np.maximum(cp * (F - K), 0.0)
    identifiable = (p - intrinsic) > 1e-10 * np.maximum(1.0, F)

    # every identifiable quote recovers sigma
    assert np.isfinite(iv[identifiable]).all()
    np.testing.assert_allclose(iv[identifiable], sigma[identifiable], rtol=1e-6, atol=1e-8)
    # and repricing error < 1e-6 across the board
    p_back = bs.price(F[identifiable], K[identifiable], T[identifiable],
                      iv[identifiable], r[identifiable], cp[identifiable])
    assert np.max(np.abs(p_back - p[identifiable])) < 1e-6
    # sanity: sweep actually exercises the solver (most quotes identifiable)
    assert identifiable.mean() > 0.9


def test_recovery_across_tenor_structure():
    # term structure of vol, ATM options, monthly tenors out to 2y
    F, K = 100.0, 100.0
    T = np.arange(1, 25) / 12.0
    sigma_true = 0.18 + 0.04 * np.exp(-2.0 * T)   # decaying short-end premium
    p = bs.price(F, K, T, sigma_true, R, +1)
    iv = implied_vol(p, F, K, T, R, +1)
    np.testing.assert_allclose(iv, sigma_true, rtol=1e-8)
