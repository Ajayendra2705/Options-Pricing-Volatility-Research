"""
Black-Scholes pricing + first-order Greeks. Forward-based core (Black-76).

Core functions take (F, K, T, sigma, r, cp):
    F     forward price of underlying for expiry T
    K     strike
    T     time to expiry in years (> 0)
    sigma Black implied volatility (> 0)
    r     continuously-compounded discount rate (discount factor = exp(-r*T))
    cp    +1 for call, -1 for put

Spot wrappers (suffix `_spot`) take (S, K, T, sigma, r, q, cp) and convert via
F = S * exp((r - q) * T), where q is a continuous dividend/carry yield.

All functions are numpy-vectorized. sigma*sqrt(T) == 0 collapses to the
discounted-intrinsic limit (needed later for IV inversion edge cases).

Conventions:
    theta  = dV/dt (calendar decay, per year; negative for long options)
    vega   = dV/dsigma (per 1.0 of vol, i.e. per 100 vol points)
    rho    = dV/dr (per 1.0 of rate)
Forward-core Greeks hold F fixed; spot Greeks hold S fixed (standard textbook).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(F, K, T, sigma):
    """Return (d1, d2, s) where s = sigma*sqrt(T) total stdev. s==0 -> d1,d2 = +/-inf sign(F-K)."""
    F, K, T, sigma = np.broadcast_arrays(
        np.asarray(F, float), np.asarray(K, float), np.asarray(T, float), np.asarray(sigma, float)
    )
    s = sigma * np.sqrt(T)
    # s == 0 limit: d1 -> +inf (F>K), -inf (F<K), 0 (F==K, N(0)=0.5 makes price collapse to 0)
    d_lim = np.where(F > K, np.inf, np.where(F < K, -np.inf, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(s > 0, (np.log(F / K) + 0.5 * s**2) / np.where(s > 0, s, 1.0), d_lim)
        d2 = d1 - s
    return d1, d2, s


def price(F, K, T, sigma, r, cp):
    """Black-76 price: exp(-r*T) * cp * (F*N(cp*d1) - K*N(cp*d2))."""
    d1, d2, _ = _d1_d2(F, K, T, sigma)
    cp = np.asarray(cp, float)
    df = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    return df * cp * (np.asarray(F, float) * norm.cdf(cp * d1) - np.asarray(K, float) * norm.cdf(cp * d2))


def delta(F, K, T, sigma, r, cp):
    """Forward delta dV/dF = exp(-r*T) * cp * N(cp*d1)."""
    d1, _, _ = _d1_d2(F, K, T, sigma)
    cp = np.asarray(cp, float)
    df = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    return df * cp * norm.cdf(cp * d1)


def gamma(F, K, T, sigma, r, cp=None):
    """Forward gamma d2V/dF2 = exp(-r*T) * phi(d1) / (F * sigma * sqrt(T)). Same for call/put."""
    d1, _, s = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = df * norm.pdf(d1) / (np.asarray(F, float) * np.where(s > 0, s, 1.0))
    return np.where(s > 0, g, 0.0)


def vega(F, K, T, sigma, r, cp=None):
    """dV/dsigma = exp(-r*T) * F * phi(d1) * sqrt(T). Same for call/put."""
    d1, _, _ = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    return df * np.asarray(F, float) * norm.pdf(d1) * np.sqrt(np.asarray(T, float))


def theta(F, K, T, sigma, r, cp):
    """Calendar decay dV/dt at fixed F: r*V - exp(-r*T)*F*phi(d1)*sigma / (2*sqrt(T))."""
    d1, _, s = _d1_d2(F, K, T, sigma)
    T = np.asarray(T, float)
    df = np.exp(-np.asarray(r, float) * T)
    v = price(F, K, T, sigma, r, cp)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = df * np.asarray(F, float) * norm.pdf(d1) * np.asarray(sigma, float) / (2.0 * np.sqrt(T))
    return np.asarray(r, float) * v - np.where(s > 0, decay, 0.0)


def rho(F, K, T, sigma, r, cp):
    """dV/dr at fixed F: only the discount factor depends on r -> rho = -T * V."""
    return -np.asarray(T, float) * price(F, K, T, sigma, r, cp)


# ---------------------------------------------------------------------------
# Second-order Greeks (Day 4). Same for call/put.
# ---------------------------------------------------------------------------


def vanna(F, K, T, sigma, r, cp=None):
    """d2V/dFdsigma = dDelta_fwd/dsigma = -exp(-r*T) * phi(d1) * d2 / sigma.

    Derivation: d(d1)/dsigma = -d2/sigma, and delta_fwd = df*cp*N(cp*d1) whose
    sigma-derivative is df*phi(d1)*d(d1)/dsigma for both cp signs.
    """
    d1, d2, s = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, float) * np.asarray(T, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -df * norm.pdf(d1) * d2 / np.where(np.asarray(sigma, float) > 0, np.asarray(sigma, float), 1.0)
    return np.where(s > 0, v, 0.0)


def volga(F, K, T, sigma, r, cp=None):
    """d2V/dsigma2 = dvega/dsigma = vega * d1 * d2 / sigma. Also called vomma."""
    d1, d2, s = _d1_d2(F, K, T, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vega(F, K, T, sigma, r) * d1 * d2 / np.where(
            np.asarray(sigma, float) > 0, np.asarray(sigma, float), 1.0
        )
    return np.where(s > 0, v, 0.0)


# ---------------------------------------------------------------------------
# Spot wrappers: S, r, q -> F = S*exp((r-q)*T). Standard textbook Greeks.
# ---------------------------------------------------------------------------


def forward(S, r, q, T):
    """Forward from spot with continuous carry: F = S * exp((r - q) * T)."""
    return np.asarray(S, float) * np.exp((np.asarray(r, float) - np.asarray(q, float)) * np.asarray(T, float))


def price_spot(S, K, T, sigma, r, q, cp):
    return price(forward(S, r, q, T), K, T, sigma, r, cp)


def delta_spot(S, K, T, sigma, r, q, cp):
    """dV/dS = cp * exp(-q*T) * N(cp*d1)."""
    d1, _, _ = _d1_d2(forward(S, r, q, T), K, T, sigma)
    cp = np.asarray(cp, float)
    return cp * np.exp(-np.asarray(q, float) * np.asarray(T, float)) * norm.cdf(cp * d1)


def gamma_spot(S, K, T, sigma, r, q, cp=None):
    """d2V/dS2 = exp(-q*T) * phi(d1) / (S * sigma * sqrt(T))."""
    d1, _, s = _d1_d2(forward(S, r, q, T), K, T, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.exp(-np.asarray(q, float) * np.asarray(T, float)) * norm.pdf(d1) / (
            np.asarray(S, float) * np.where(s > 0, s, 1.0)
        )
    return np.where(s > 0, g, 0.0)


def vega_spot(S, K, T, sigma, r, q, cp=None):
    """dV/dsigma = S * exp(-q*T) * phi(d1) * sqrt(T). Identical to forward-core vega."""
    return vega(forward(S, r, q, T), K, T, sigma, r)


def theta_spot(S, K, T, sigma, r, q, cp):
    """Textbook theta at fixed S:
    -S*e^{-qT}*phi(d1)*sigma/(2*sqrt(T)) - cp*r*K*e^{-rT}*N(cp*d2) + cp*q*S*e^{-qT}*N(cp*d1)
    """
    S, K, T, sigma, r, q = (np.asarray(x, float) for x in (S, K, T, sigma, r, q))
    cp = np.asarray(cp, float)
    d1, d2, s = _d1_d2(forward(S, r, q, T), K, T, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    decay = np.where(s > 0, decay, 0.0)
    return (
        -decay
        - cp * r * K * np.exp(-r * T) * norm.cdf(cp * d2)
        + cp * q * S * np.exp(-q * T) * norm.cdf(cp * d1)
    )


def rho_spot(S, K, T, sigma, r, q, cp):
    """dV/dr at fixed S: cp * K * T * exp(-r*T) * N(cp*d2)."""
    K, T, r = np.asarray(K, float), np.asarray(T, float), np.asarray(r, float)
    cp = np.asarray(cp, float)
    _, d2, _ = _d1_d2(forward(S, r, q, T), K, T, sigma)
    return cp * K * T * np.exp(-r * T) * norm.cdf(cp * d2)


def vanna_spot(S, K, T, sigma, r, q, cp=None):
    """d2V/dSdsigma = dDelta_spot/dsigma = -exp(-q*T) * phi(d1) * d2 / sigma."""
    d1, d2, s = _d1_d2(forward(S, r, q, T), K, T, sigma)
    sigma = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -np.exp(-np.asarray(q, float) * np.asarray(T, float)) * norm.pdf(d1) * d2 / np.where(
            sigma > 0, sigma, 1.0
        )
    return np.where(s > 0, v, 0.0)


def volga_spot(S, K, T, sigma, r, q, cp=None):
    """d2V/dsigma2 at fixed S: vega * d1 * d2 / sigma (vega is spot==forward)."""
    return volga(forward(S, r, q, T), K, T, sigma, r)
