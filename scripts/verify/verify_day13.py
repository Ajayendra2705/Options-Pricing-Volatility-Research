from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Day 13 independent verify.
1. svi_params_joint.parquet: FD-only Durrleman g >= 0 for every fitted slice
   (no analytic derivatives from the codebase).
2. Calendar monotonicity re-checked on a DENSER, OFF-NODE grid (4001 pts,
   irrational offset) so sub-node violations can't hide.
3. Economic anchor: normalized call prices C/F = Black(1, e^k, T, iv) must be
   non-decreasing in T at fixed forward moneyness (model-free calendar test,
   bypasses total-variance algebra entirely).
4. results/arb_violations.json consistent with a fresh recomputation.
"""
import sys, json
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from math import erf

root = ROOT


def w_svi(k, a, b, rho, m, s):
    d = k - m
    return a + b * (rho * d + np.sqrt(d**2 + s**2))


def g_fd(k, prm, h=1e-5):
    f = lambda x: w_svi(x, *prm)
    w = f(k)
    wp = (f(k + h) - f(k - h)) / (2 * h)
    wpp = (f(k + h) - 2 * w + f(k - h)) / h**2
    return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x) / np.sqrt(2.0)))


def call_norm(k, w):
    """Undiscounted normalized Black call C/F with total variance w, strike e^k."""
    s = np.sqrt(w)
    d1 = (-k + w / 2) / s
    return norm_cdf(d1) - np.exp(k) * norm_cdf(d1 - s)


fits = pd.read_parquet(root + "/data/processed/svi_params_joint.parquet")
ok = fits[fits["fit_ok"] == True].copy()
print(f"fitted slices: {len(ok)}")

# 1. FD-only butterfly on off-node grid
kd = np.linspace(-1.5, 1.5, 2001) + 1e-4 * np.pi
gmins = []
for f in ok.itertuples():
    prm = (f.a, f.b, f.rho, f.m, f.sigma)
    gmins.append(g_fd(kd, prm).min())
print(f"1. FD-g min across {len(ok)} slices: {min(gmins):+.5f} (all >= 0: {min(gmins) >= 0})")
assert min(gmins) >= 0

# 2+3. calendar on dense off-node grid + call-price monotonicity
#
# Domain (Day 32): the pair's QUOTED log-moneyness overlap, not a fixed +-1.5.
# Past the quoted strikes an SVI slice is extrapolation with linear wings, the
# fit does not constrain it there, and no calendar spread is tradeable there
# either — scanning it measured the extrapolation, not the surface. The grid is
# still dense and off-node (irrational offset) so sub-node violations inside the
# claimed domain cannot hide.
worst_w, worst_c, n_pairs = -np.inf, -np.inf, 0
for date, g in ok.groupby("date"):
    g = g.sort_values("T")
    rows = list(g.itertuples())
    for i in range(len(rows) - 1):
        s, l = rows[i], rows[i + 1]
        lo, hi = max(s.k_lo, l.k_lo), min(s.k_hi, l.k_hi)
        if not hi > lo:                       # disjoint quoted strikes
            continue
        n_pairs += 1
        kc = np.linspace(lo, hi, 4001) + np.sqrt(2) * 1e-7
        prm_s = (s.a, s.b, s.rho, s.m, s.sigma)
        prm_l = (l.a, l.b, l.rho, l.m, l.sigma)
        w1, w2 = w_svi(kc, *prm_s), w_svi(kc, *prm_l)
        worst_w = max(worst_w, (w1 - w2).max())
        c1, c2 = call_norm(kc, w1), call_norm(kc, w2)
        worst_c = max(worst_c, (c1 - c2).max())
print(f"2. calendar off-node 4001-pt: {n_pairs} pairs, max w-decrease {worst_w:+.2e} "
      f"(clean: {worst_w <= 0})")
print(f"3. call-price monotonicity: max C(T_short)-C(T_long) = {worst_c:+.2e} "
      f"(clean: {worst_c <= 1e-12})")
assert worst_w <= 0 and worst_c <= 1e-12

# 4. json consistency
rep = json.load(open(root + "/results/arb_violations.json"))
assert rep["n_slices_fitted"] == len(ok)
assert rep["butterfly"]["n_violations"] == int((~ok["arb_free"].astype(bool)).sum()) == 0
assert rep["calendar"]["n_pairs_checked"] == n_pairs
assert rep["calendar"]["n_pairs_violated"] == 0
assert abs(rep["rmse_iv_median"] - ok["rmse_iv"].median()) < 1e-15
assert abs(rep["butterfly"]["min_g_across_slices"] - ok["min_g"].min()) < 1e-15
print(f"4. arb_violations.json consistent: {rep['n_slices_fitted']} slices, "
      f"{rep['calendar']['n_pairs_checked']} pairs, median RMSE "
      f"{rep['rmse_iv_median']*100:.2f} volpts")
print("DAY 13 INDEPENDENT CHECKS PASSED")
