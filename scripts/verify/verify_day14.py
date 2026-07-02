from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Day 14 independent verify (updated for the hardened interpolation spec:
price-space interior, flat-IV short extrap, flat-w long extrap).
1. Inline reimplementation of the T-interpolation (linear in normalized
   OTM option price at fixed k, inverted via scipy brentq — NOT the
   project's solver), ln-F-linear forwards, cross-checked against
   VolSurface at random (k, T) points — catches wiring bugs in assemble.py.
2. FD Durrleman g on 51 interpolated T-slices per date, off-node k grid
   (denser than QC's 21).
3. Economic anchor: Breeden-Litzenberger density of INTERPOLATED slices —
   nonneg, mass ~1, mean ~1 (normalized forward measure).
4. surface_qc.json vs fresh inline recompute of market residuals.
"""
import sys, json
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from math import erf

from src.surface.assemble import build_surfaces
from src.surface.svi import otm_side

root = ROOT
rng = np.random.default_rng(42)


def w_svi(k, a, b, rho, m, s):
    d = np.asarray(k, float) - m
    return a + b * (rho * d + np.sqrt(d**2 + s**2))


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x) / np.sqrt(2.0)))


fits = pd.read_parquet(root + r"\data\processed\svi_params_joint.parquet")
fwd = pd.read_parquet(root + r"\data\processed\forwards.parquet")
market = pd.read_parquet(root + r"\data\processed\iv_surface.parquet")
surfaces = build_surfaces(fits, fwd)
ok = fits[fits["fit_ok"] == True]

# 1. inline interp vs VolSurface at random points
def black_norm(k, w, cp):
    s = np.sqrt(w)
    d1 = -k / s + s / 2.0
    return cp * (norm_cdf(cp * d1) - np.exp(k) * norm_cdf(cp * (d1 - s)))


def w_inline(date, k, T):
    from scipy.optimize import brentq
    g = ok[ok["date"] == date].sort_values("T")
    Ts = g["T"].to_numpy()
    prms = [(f.a, f.b, f.rho, f.m, f.sigma) for f in g.itertuples()]
    if T <= Ts[0]:
        return w_svi(k, *prms[0]) * T / Ts[0]
    if T >= Ts[-1]:
        return w_svi(k, *prms[-1])                      # flat-w long extrap
    i = np.searchsorted(Ts, T, side="right") - 1
    lam = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
    w1, w2 = w_svi(k, *prms[i]), w_svi(k, *prms[i + 1])
    cp = -1 if k < 0 else 1
    p = (1 - lam) * black_norm(k, w1, cp) + lam * black_norm(k, w2, cp)
    lo, hi = min(w1, w2), max(w1, w2)
    f = lambda w: black_norm(k, w, cp) - p
    if f(lo) * f(hi) > 0:                               # underflowed wing
        return (1 - lam) * w1 + lam * w2
    return brentq(f, lo, hi, xtol=1e-15)


def F_inline(date, T):
    # Day-15 spec: forward curve uses ALL implied-forward nodes for the date,
    # including expiries whose vol slice failed to fit
    g = fwd[fwd["date"] == date].sort_values("T")
    Ts, lnF = g["T"].to_numpy(), np.log(g["F"].to_numpy())
    i = min(max(np.searchsorted(Ts, T, side="right") - 1, 0), len(Ts) - 2)
    return np.exp(lnF[i] + (lnF[i + 1] - lnF[i]) / (Ts[i + 1] - Ts[i]) * (T - Ts[i]))


worst = 0.0
for date, vs in surfaces.items():
    for _ in range(200):
        k = rng.uniform(-1.2, 1.2)
        T = rng.uniform(0.4 * vs.Ts[0], 1.5 * vs.Ts[-1])
        worst = max(worst, abs(vs.w(k, T) - w_inline(date, k, T)))
        worst = max(worst, abs(vs.forward(T) - F_inline(date, T)) / F_inline(date, T))
print(f"1. inline vs VolSurface, 1000 random (k,T): worst discrepancy {worst:.2e}")
# two independent root-finders (vectorized Newton vs scipy brentq) agree to
# solver precision; 1e-9 is far below any economic scale
assert worst < 1e-9

# 2. FD g on 51 interpolated slices per date, off-node grid
kd = np.linspace(-1.0, 1.0, 1601) + 1e-4 * np.pi
h = 1e-5
worst_g = np.inf
for date, vs in surfaces.items():
    for T in np.linspace(0.5 * vs.Ts[0], 1.25 * vs.Ts[-1], 51):
        w = vs.w(kd, T)
        wp = (vs.w(kd + h, T) - vs.w(kd - h, T)) / (2 * h)
        wpp = (vs.w(kd + h, T) - 2 * w + vs.w(kd - h, T)) / h**2
        g = (1 - kd * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2
        worst_g = min(worst_g, g.min())
print(f"2. FD-g on 255 interpolated slices: min {worst_g:+.4f} (all >= 0: {worst_g >= 0})")
assert worst_g >= 0

# 3. BL density on interpolated slices (3 random T per date)
worst_dmin, worst_mass, worst_mean = 0.0, 0.0, 0.0
kk = np.linspace(-1.5, 1.5, 6001)
Kx = np.exp(kk)
for date, vs in surfaces.items():
    for T in np.linspace(vs.Ts[0] * 1.1, vs.Ts[-1] * 0.95, 3):
        s = np.sqrt(vs.w(kk, T))
        d1 = (-kk + s**2 / 2) / s
        C = norm_cdf(d1) - Kx * norm_cdf(d1 - s)      # normalized undiscounted call
        dens = np.gradient(np.gradient(C, Kx), Kx)
        worst_dmin = min(worst_dmin, dens.min())
        worst_mass = max(worst_mass, abs(np.trapezoid(dens, Kx) - 1))
        worst_mean = max(worst_mean, abs(np.trapezoid(Kx * dens, Kx) - 1))
print(f"3. BL density, 15 interpolated slices: min {worst_dmin:+.2e}, "
      f"|mass-1| max {worst_mass:.2e}, |mean-1| max {worst_mean:.2e}")
assert worst_dmin > -1e-8 and worst_mass < 5e-3 and worst_mean < 5e-3

# 4. qc json vs fresh residual recompute
qc = json.load(open(root + r"\results\surface_qc.json"))
assert qc["n_dates"] == len(surfaces) == 5
for d in qc["dates"]:
    date = pd.Timestamp(d["date"])
    vs = surfaces[date]
    errs = []
    for expiry, g in market[market["date"] == date].groupby("expiry"):
        sl = otm_side(g)
        T = float(sl["T"].iloc[0])
        errs.append(vs.iv(sl["log_moneyness"].to_numpy(), T) - sl["iv"].to_numpy())
    e = np.concatenate(errs)
    assert abs(np.sqrt(np.mean(e**2)) - d["rmse_iv"]) < 1e-14, d["date"]
    assert d["n_market_quotes"] == e.size
print(f"4. surface_qc.json residuals match inline recompute on all {qc['n_dates']} dates "
      f"(median RMSE {qc['rmse_iv_median']*100:.2f} volpts)")
print("DAY 14 INDEPENDENT CHECKS PASSED")
