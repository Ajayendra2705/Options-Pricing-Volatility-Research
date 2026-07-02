from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Day 4 independent verify: complex-step derivatives (exact to machine eps).

Rebuild Black-76 with complex-capable N(x) = 0.5*(1+erf(x/sqrt(2))) via
scipy.special.erf, then complex-step differentiate: f'(x) = Im f(x+ih)/h,
h = 1e-20 -> no subtractive cancellation, ~machine precision.
Cross-check analytic vanna/volga (and delta/vega/gamma) against it.
"""
import sys
sys.path.insert(0, ROOT)
import numpy as np
from scipy.special import erf
from src.greeks import black_scholes as bs

SQRT2 = np.sqrt(2.0)

def N(x):        # complex-capable normal cdf
    return 0.5 * (1.0 + erf(x / SQRT2))

def price_c(F, K, T, sigma, r, cp):   # complex-capable Black-76
    s = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    return np.exp(-r * T) * cp * (F * N(cp * d1) - K * N(cp * d2))

def cstep(f, x, h=1e-20):
    return (f(x + 1j * h)).imag / h

H = 1e-20
r, Kk = 0.03, 100.0
worst = {"delta": 0, "vega": 0, "gamma": 0, "vanna": 0, "volga": 0}
for F in (80.0, 100.0, 120.0):
    for T in (0.05, 0.5, 2.0):
        for sig in (0.10, 0.30, 0.80):
            for cp in (+1, -1):
                # first order via complex step of price
                d_cs = cstep(lambda x: price_c(x, Kk, T, sig, r, cp), F)
                v_cs = cstep(lambda x: price_c(F, Kk, T, x, r, cp), sig)
                # second order via complex step of complex-step-free analytic?
                # -> complex step of *rebuilt* delta/vega (independent impl)
                delta_c = lambda F_, s_: np.exp(-r * T) * cp * N(cp * ((np.log(F_ / Kk) + 0.5 * s_**2 * T) / (s_ * np.sqrt(T))))
                g_cs = cstep(lambda x: delta_c(x, sig), F)
                vanna_cs = cstep(lambda x: delta_c(F, x), sig)
                vega_c = lambda s_: cstep(lambda x: price_c(F, Kk, T, x, r, cp), s_)  # not complex-chainable
                # volga: central FD of complex-step vega (cs vega is exact -> FD is clean)
                hh = 1e-6 * sig
                volga_ref = (cstep(lambda x: price_c(F, Kk, T, x, r, cp), sig + hh)
                             - cstep(lambda x: price_c(F, Kk, T, x, r, cp), sig - hh)) / (2 * hh)

                def relerr(a, b):
                    return abs(a - b) / max(1e-12, abs(b))

                worst["delta"] = max(worst["delta"], relerr(bs.delta(F, Kk, T, sig, r, cp), d_cs))
                worst["vega"] = max(worst["vega"], relerr(bs.vega(F, Kk, T, sig, r), v_cs))
                worst["gamma"] = max(worst["gamma"], relerr(bs.gamma(F, Kk, T, sig, r), g_cs))
                worst["vanna"] = max(worst["vanna"], relerr(bs.vanna(F, Kk, T, sig, r), vanna_cs))
                worst["volga"] = max(worst["volga"], relerr(bs.volga(F, Kk, T, sig, r), volga_ref))

for k, v in worst.items():
    print(f"{k:6s} worst rel err vs complex-step: {v:.3e}")

# analytic identity check: vanna == vega/F * (1 - d1/(sigma*sqrt(T)))
ok = True
for F in (80.0, 95.0, 100.0, 110.0, 130.0):
    for sig in (0.1, 0.4):
        T = 0.6
        s = sig * np.sqrt(T)
        d1 = (np.log(F / Kk) + 0.5 * s * s) / s
        alt = bs.vega(F, Kk, T, sig, r) / F * (1.0 - d1 / s)
        ok &= np.isclose(bs.vanna(F, Kk, T, sig, r), alt, rtol=1e-12)
print("vanna alt-identity (vega/F*(1-d1/s)):", "OK" if ok else "FAIL")
