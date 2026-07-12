from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Day 12 verify.
1. svi_params_constrained.parquet: FD-only g >= 0 on all fitted slices;
   log json consistent with parquet.
2. Vogt hard case rerun: constrained params -> independent BL density
   check (nonneg, mass ~1, mean ~F) + RMSE vs quotes recomputed inline.
"""
import sys, json
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from src.surface.svi import SVIParams, svi_iv, svi_total_variance
from src.surface.no_arb import fit_svi_constrained
from src.greeks import black_scholes as bs

root = ROOT

def g_fd(k, p, h=1e-5):
    w = svi_total_variance(k, p)
    wp = (svi_total_variance(k + h, p) - svi_total_variance(k - h, p)) / (2 * h)
    wpp = (svi_total_variance(k + h, p) - 2 * w + svi_total_variance(k - h, p)) / h**2
    return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2

# 1. parquet + log consistency
fits = pd.read_parquet(root + "/data/processed/svi_params_constrained.parquet")
log = json.load(open(root + "/results/svi_butterfly_log.json"))
ok = fits[fits["fit_ok"] == True]
k = np.linspace(-1.5, 1.5, 2001)
gmins = [g_fd(k, (f.a, f.b, f.rho, f.m, f.sigma)).min() for f in ok.itertuples()]
print(f"parquet slices: {len(ok)}, FD-g min across all: {min(gmins):+.5f} (all >= 0: {min(gmins) >= 0})")
assert min(gmins) >= 0
assert log["n_fitted"] == len(ok) and log["n_post_violations"] == 0
assert len(log["slices"]) == len(ok)
for ls, f in zip(log["slices"], ok.itertuples()):
    assert abs(ls["rmse_constrained"] - f.rmse_iv) < 1e-12
print("log json consistent with parquet")

# 2. Vogt hard case, end to end, independent verdict
VOGT = SVIParams(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
K_GRID = np.linspace(-0.9, 0.9, 31); T = 1.0
iv_mkt = svi_iv(K_GRID, T, VOGT)
p, rep = fit_svi_constrained(K_GRID, iv_mkt, T)
print(f"vogt refit: method={rep['method']} arb_free={rep['arb_free']} min_g={rep['min_g']:+.2e}")
# independent: BL density of the constrained slice
kk = np.linspace(-1.3, 1.3, 5001); Kx = np.exp(kk)
ivc = np.sqrt(np.maximum(svi_total_variance(kk, p), 1e-12) / T)
C = bs.price(1.0, Kx, T, ivc, 0.0, 1)
dens = np.gradient(np.gradient(C, Kx), Kx)
print(f"constrained-Vogt density: min={dens.min():+.2e} mass={np.trapezoid(dens, Kx):.4f} "
      f"mean={np.trapezoid(Kx * dens, Kx):.4f}")
assert dens.min() > -1e-6
# inline RMSE vs quotes
rmse = np.sqrt(np.mean((np.sqrt(np.maximum(svi_total_variance(K_GRID, p), 0) / T) - iv_mkt) ** 2))
print(f"constrained-Vogt RMSE recomputed: {rmse * 100:.3f} volpts (report {rep['rmse_iv'] * 100:.3f})")
assert abs(rmse - rep["rmse_iv"]) < 1e-12
print("DAY 12 INDEPENDENT CHECKS PASSED")
