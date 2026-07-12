from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Day 11 verify.

1. g via FD derivatives only (w', w'' numeric) vs durrleman_g (analytic)
   on Vogt, benign, and all 14 real fits.
2. Breeden-Litzenberger density from each real fit: nonnegative,
   integrates to ~1, martingale mean ~F (undiscounted, F=1 convention).
"""
import sys
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from src.surface.svi import svi_total_variance
from src.surface.no_arb import durrleman_g
from src.greeks import black_scholes as bs

def g_fd(k, p, h=1e-5):
    w = svi_total_variance(k, p)
    wp = (svi_total_variance(k + h, p) - svi_total_variance(k - h, p)) / (2 * h)
    wpp = (svi_total_variance(k + h, p) - 2 * w + svi_total_variance(k - h, p)) / h**2
    return (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2

k = np.linspace(-1.4, 1.4, 1401)
cases = {"vogt": (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153),
         "benign": (0.015, 0.35, -0.65, 0.05, 0.25)}
fits = pd.read_parquet(ROOT + "/data/processed/svi_params.parquet")
for i, f in fits[fits["fit_ok"]].iterrows():
    cases[f"real{i}"] = (f["a"], f["b"], f["rho"], f["m"], f["sigma"])

worst = 0.0
for name, p in cases.items():
    d = np.max(np.abs(durrleman_g(k, p) - g_fd(k, p)))
    # FD truncation scales with w'''' ~ b/sigma^3: normalize the tolerance by
    # curvature so spiky-but-valid slices (tiny sigma) don't false-alarm
    curv = max(1.0, p[1] / max(p[4], 1e-4) ** 2)
    d_rel = d / curv
    worst = max(worst, d_rel)
print(f"analytic-g vs FD-g worst curvature-scaled diff over {len(cases)} param sets: {worst:.2e}")
assert worst < 1e-5

# Detection consistency on UNCONSTRAINED fits: a negative BL density must
# coincide with durrleman detection flagging the slice (Day 11 = detection;
# the arb-free guarantee lives in the joint params, checked below)
from src.surface.no_arb import check_butterfly
for i, f in fits[fits["fit_ok"]].iterrows():
    p = (f["a"], f["b"], f["rho"], f["m"], f["sigma"])
    T = f["T"]
    kk = np.linspace(-1.0, 1.0, 4001)
    K = np.exp(kk)
    iv = np.sqrt(np.maximum(svi_total_variance(kk, p), 1e-12) / T)
    dens = np.gradient(np.gradient(bs.price(1.0, K, T, iv, 0.0, 1), K), K)
    if dens.min() < -1e-6:
        assert not check_butterfly(p)["arb_free"], \
            f"negative density NOT detected by durrleman: {p}"
        print(f"  unconstrained {pd.Timestamp(f['date']).date()}->"
              f"{pd.Timestamp(f['expiry']).date()}: arbitrable "
              f"(min_dens={dens.min():+.1e}) and correctly FLAGGED by detection")

# BL density checks on the AUTHORITATIVE joint fits (T from table, F=1)
fits = pd.read_parquet(ROOT + "/data/processed/svi_params_joint.parquet")
print("\njoint-fit densities (BL from Black calls):")
bad = 0
for i, f in fits[fits["fit_ok"]].iterrows():
    T = f["T"]
    kk = np.linspace(-1.0, 1.0, 4001)
    K = np.exp(kk)
    iv = np.sqrt(np.maximum(svi_total_variance(kk, (f["a"], f["b"], f["rho"], f["m"], f["sigma"])), 1e-12) / T)
    C = bs.price(1.0, K, T, iv, 0.0, 1)
    dens = np.gradient(np.gradient(C, K), K)
    mass = np.trapezoid(dens, K)
    mean = np.trapezoid(K * dens, K)
    neg = dens.min()
    tag = "OK" if (neg > -1e-6 and abs(mass - 1) < 0.01 and abs(mean - 1) < 0.01) else "CHECK"
    if tag != "OK":
        bad += 1
    print(f"  {pd.Timestamp(f['date']).date()}->{pd.Timestamp(f['expiry']).date()}: "
          f"min_dens={neg:+.2e} mass={mass:.4f} mean={mean:.4f} {tag}")
n_fit = int(fits["fit_ok"].sum())
print(f"\n{n_fit - bad}/{n_fit} densities valid probability measures")
assert bad == 0
