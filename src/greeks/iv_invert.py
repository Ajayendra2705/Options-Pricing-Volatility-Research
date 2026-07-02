"""
Implied-vol inversion. Forward-based (Black-76), robust: Newton with
Brent bracket fallback.

Core: implied_vol(price, F, K, T, r, cp) -> sigma (nan when no-arb bounds
violated, 0.0 when price sits at discounted intrinsic).
Spot wrapper handles carry via F = S*exp((r-q)*T).

Method:
  1. Undiscount to forward premium p = price / df, check no-arb bounds:
       max(cp*(F-K), 0) <= p <= F (call) / K (put)
  2. Newton from sigma0 = 0.5 using analytic vega (fast: quadratic conv).
  3. If Newton stalls (tiny vega, step out of [lo, hi], no convergence),
     fall back to Brent on the bracketed price residual — guaranteed,
     since Black price is strictly increasing in sigma inside the bounds.

Vectorized via a scalar core + np.frompyfunc loop (chains are ~1e3-1e5 rows;
fine, and keeps the root-finder logic readable).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from src.greeks import black_scholes as bs

SIGMA_LO = 1e-9
SIGMA_HI = 5.0
PRICE_TOL = 1e-12   # convergence on undiscounted price residual (per unit F~O(100) -> plenty below 1e-6 target)
MAX_NEWTON = 20


def _implied_vol_scalar(price, F, K, T, r, cp):
    """Scalar Black-76 IV. Returns nan outside no-arb bounds, 0.0 at intrinsic."""
    if not (np.isfinite(price) and F > 0 and K > 0 and T > 0):
        return np.nan
    df = np.exp(-r * T)
    p = price / df                       # undiscounted forward premium
    intrinsic = max(cp * (F - K), 0.0)
    upper = F if cp > 0 else K           # sigma -> inf limit
    # loose eps only for *rejecting* arb violations (real quotes are noisy)
    eps = 1e-9 * max(1.0, F, K)
    if p < intrinsic - eps or p > upper + eps:
        return np.nan
    if p <= intrinsic:                   # at/below intrinsic (within eps): zero vol
        return 0.0
    p = min(p, upper)                    # clamp roundoff above the sigma->inf limit

    def resid(sig):
        return bs.price(F, K, T, sig, 0.0, cp) - p   # r=0: already undiscounted

    # --- Newton: converge on the sigma step (quadratic near the root) ---
    sig = 0.5
    for _ in range(MAX_NEWTON):
        diff = resid(sig)
        if diff == 0.0:
            return float(sig)
        v = bs.vega(F, K, T, sig, 0.0)
        if v < 1e-16:                     # flat wing: Newton useless here
            break
        step = diff / v
        new = sig - step
        if not (SIGMA_LO < new < SIGMA_HI):
            break
        sig = new
        if abs(step) < 1e-12 * max(1.0, sig):
            return float(sig)

    # --- Brent fallback (guaranteed: Black price strictly increasing in sigma) ---
    lo, hi = SIGMA_LO, SIGMA_HI
    if resid(lo) >= 0:                    # premium indistinguishable from intrinsic
        return 0.0
    if resid(hi) < 0:                     # beyond 500% vol: no solution in range
        return np.nan
    return float(brentq(resid, lo, hi, xtol=1e-16, rtol=8.9e-16, maxiter=200))


_ufunc = np.frompyfunc(_implied_vol_scalar, 6, 1)


def implied_vol(price, F, K, T, r, cp):
    """Vectorized Black-76 implied vol. nan = no-arb violation / no solution."""
    out = np.asarray(_ufunc(price, F, K, T, r, cp), dtype=float)
    return out if out.ndim else float(out)


def implied_vol_spot(price, S, K, T, r, q, cp):
    """Spot-quoted IV with continuous carry q: converts to forward, same core."""
    return implied_vol(price, bs.forward(S, r, q, T), K, T, r, cp)
