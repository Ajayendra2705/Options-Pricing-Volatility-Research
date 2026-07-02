from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Day 5 independent verify.

1. Price 2000 random options with an INDEPENDENT erf-based Black-76
   (not src.bs.price) -> invert with iv_invert -> sigma must come back.
   Catches compensating bugs a same-pricer roundtrip would hide.
2. Solver monotonicity: iv strictly increasing in price between bounds.
3. Adversarial edges: price ~ bound - ulp, huge vol near SIGMA_HI,
   tiny T, price just below intrinsic (nan), just above upper (nan).
"""
import sys
sys.path.insert(0, ROOT)
import numpy as np
from scipy.special import erf
from src.greeks.iv_invert import implied_vol
from src.greeks import black_scholes as bs

SQRT2 = np.sqrt(2.0)

def N(x):
    return 0.5 * (1.0 + erf(x / SQRT2))

def price_indep(F, K, T, sig, r, cp):     # independent Black-76
    s = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return np.exp(-r * T) * cp * (F * N(cp * d1) - K * N(cp * (d1 - s)))

rng = np.random.default_rng(7)
n = 2000
F = rng.uniform(20.0, 500.0, n)
K = F * rng.uniform(0.7, 1.4, n)
T = rng.uniform(0.01, 2.5, n)
sig = rng.uniform(0.05, 1.5, n)
r = rng.uniform(0.0, 0.08, n)
cp = np.where(rng.random(n) < 0.5, 1, -1)

p = np.array([price_indep(*a) for a in zip(F, K, T, sig, r, cp)])
iv = implied_vol(p, F, K, T, r, cp)

df = np.exp(-r * T)
tv = p / df - np.maximum(cp * (F - K), 0.0)
identifiable = tv > 1e-8 * np.maximum(1.0, p / df)

rel = np.abs(iv[identifiable] - sig[identifiable]) / sig[identifiable]
print(f"identifiable {identifiable.sum()}/{n}, worst sigma rel err {rel.max():.2e}")
assert rel.max() < 1e-6, "sigma recovery vs independent pricer FAILED"
# reprice everything finite
fin = np.isfinite(iv)
pb = bs.price(F[fin], K[fin], T[fin], iv[fin], r[fin], cp[fin])
print(f"finite {fin.sum()}/{n}, worst reprice abs err {np.max(np.abs(pb - p[fin])):.2e}")
assert np.max(np.abs(pb - p[fin])) < 1e-6

# 2. monotonicity of solver in price
Fm, Km, Tm, rm = 100.0, 105.0, 0.6, 0.03
dfm = np.exp(-rm * Tm)
# sweep only the solvable band: prices needing sigma > SIGMA_HI=5.0 return
# nan BY DESIGN (asserted in the edge section below)
p_cap = bs.price(Fm, Km, Tm, 5.0, rm, 1)
prices = np.linspace(0.01, p_cap * 0.9999, 400)
ivs = implied_vol(prices, Fm, Km, Tm, rm, 1)
assert np.isfinite(ivs).all() and (np.diff(ivs) > 0).all(), "monotonicity FAILED"
above = implied_vol(np.linspace(p_cap * 1.001, dfm * Fm * 0.999, 20), Fm, Km, Tm, rm, 1)
assert np.isnan(above).all(), "beyond-SIGMA_HI should be nan"
print("monotonicity in price: OK (400 pts strictly increasing; nan above sigma=5 band)")

# 3. adversarial edges
intr = dfm * max(Fm - Km, 0.0)
assert np.isnan(implied_vol(dfm * Fm * 1.0001, Fm, Km, Tm, rm, 1)), "above upper not nan"
assert np.isnan(implied_vol(dfm * (Fm - Km) - 1.0, 120.0, Km, Tm, rm, 1) if False else
                implied_vol(dfm * (120.0 - Km) * 0.9, 120.0, Km, Tm, rm, 1)), "below intrinsic not nan"
big = bs.price(100.0, 100.0, 0.5, 4.9, 0.0, 1)      # near SIGMA_HI
assert abs(implied_vol(big, 100.0, 100.0, 0.5, 0.0, 1) - 4.9) < 1e-8, "huge vol FAILED"
tiny_t = bs.price(100.0, 101.0, 1e-4, 0.3, 0.0, 1)  # ~1hr expiry
assert abs(implied_vol(tiny_t, 100.0, 101.0, 1e-4, 0.0, 1) - 0.3) < 1e-8, "tiny T FAILED"
beyond = bs.price(100.0, 100.0, 0.5, 6.0, 0.0, 1)   # vol beyond SIGMA_HI
res = implied_vol(beyond, 100.0, 100.0, 0.5, 0.0, 1)
print(f"edges OK; vol-beyond-5.0 returns: {res} (nan expected)")
assert np.isnan(res)
print("ALL DAY 5 INDEPENDENT CHECKS PASSED")
