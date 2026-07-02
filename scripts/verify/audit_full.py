from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Full-project audit — layers not covered by per-day verify scripts.

A. Cleaning invariants: chain_clean satisfies every filter's guarantee;
   counts agree with results/data_quality.json.
B. IV surface: independent brentq inversion (independent erf pricer) of a
   200-row random sample must match stored iv; status labels honest.
C. Cross-artifact spine: raw -> clean -> forwards -> iv -> svi -> joint ->
   qc all cover the same slices, no orphans.
D. Results jsons coherent with each other + PLAN deliverables all present.
"""
import sys, json
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.special import erf

root = ROOT
rng = np.random.default_rng(3)

raw = pd.read_parquet(root + r"\data\raw\aapl_options.parquet")
chain = pd.read_parquet(root + r"\data\processed\chain_clean.parquet")
fwd = pd.read_parquet(root + r"\data\processed\forwards.parquet")
surf = pd.read_parquet(root + r"\data\processed\iv_surface.parquet")
svi = pd.read_parquet(root + r"\data\processed\svi_params.parquet")
joint = pd.read_parquet(root + r"\data\processed\svi_params_joint.parquet")
dq = json.load(open(root + r"\results\data_quality.json"))
arb = json.load(open(root + r"\results\arb_violations.json"))
qc = json.load(open(root + r"\results\surface_qc.json"))
bfly = json.load(open(root + r"\results\svi_butterfly_log.json"))

# --- A. cleaning invariants ---------------------------------------------------
assert (chain["bid"] > 0).all(), "zero/neg bid survived"
assert (chain["ask"] > chain["bid"]).all(), "crossed/locked survived"
spread_pct = (chain["ask"] - chain["bid"]) / chain["mid"]
assert (spread_pct <= 0.5 + 1e-12).all(), "wide spread survived"
assert (chain["T"] > 0).all(), "expired survived"
assert chain["option_type"].isin(["C", "P"]).all()
assert not chain[["date", "expiry", "strike", "option_type"]].duplicated().any()
c = dq["cleaning"]
assert c["total_clean"] == len(chain), (c["total_clean"], len(chain))
assert c["total_rows"] == len(raw)
assert c["total_rows"] - c["total_dropped"] == len(chain)
drop = 1 - len(chain) / len(raw)
print(f"A. cleaning: {len(raw)} raw -> {len(chain)} clean ({drop:.1%} drop, json agrees); "
      f"all invariants hold")

# --- B. independent IV inversion of stored surface ----------------------------
SQRT2 = np.sqrt(2.0)
N = lambda x: 0.5 * (1.0 + erf(x / SQRT2))

def price_indep(F, K, T, sig, r, cp):
    s = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    return np.exp(-r * T) * cp * (F * N(cp * d1) - K * N(cp * (d1 - s)))

ok_rows = surf[surf["status"] == "ok"]
sample = ok_rows.sample(200, random_state=3)
worst = 0.0
for row in sample.itertuples():
    cp = 1 if row.option_type == "C" else -1
    f = lambda s: price_indep(row.F, row.strike, row.T, s, row.r, cp) - row.mid
    iv_ind = brentq(f, 1e-7, 6.0, xtol=1e-14)
    worst = max(worst, abs(iv_ind - row.iv))
print(f"B. iv surface: 200-row independent brentq re-inversion, worst |dIV| {worst:.2e}")
assert worst < 1e-9
# status honesty: below_intrinsic rows really are below discounted intrinsic
bi = surf[surf["status"] == "below_intrinsic"]
cpv = np.where(bi["option_type"] == "C", 1, -1)
intr = np.exp(-bi["r"] * bi["T"]) * np.maximum(cpv * (bi["F"] - bi["strike"]), 0)
assert (bi["mid"] <= intr + 1e-12).all()
n_ok = (surf["status"] == "ok").sum()
print(f"   status: {n_ok}/{len(surf)} ok ({n_ok/len(surf):.1%}), "
      f"{len(bi)} below_intrinsic all verified genuinely below intrinsic")

# --- C. cross-artifact spine ---------------------------------------------------
sl = lambda df: set(map(tuple, df[["date", "expiry"]].drop_duplicates().values))
assert sl(chain) == sl(fwd) == sl(surf) == sl(svi) == sl(joint), "slice spine broken"
assert len(surf) == len(chain), "quote count changed clean -> iv"
ok_joint = joint[joint["fit_ok"] == True]
assert len(ok_joint) == arb["n_slices_fitted"] == qc["n_slices_total"] == bfly["n_fitted"]
# forwards internally consistent: df = exp(-rT)
assert np.allclose(fwd["df"], np.exp(-fwd["r"] * fwd["T"]), atol=1e-12)
# joint params within sane SVI ranges
assert (ok_joint["b"] > 0).all() and (ok_joint["rho"].abs() < 1).all() \
    and (ok_joint["sigma"] > 0).all()
print(f"C. spine: {len(sl(chain))} slices consistent across all 5 artifacts; "
      f"{len(ok_joint)} fitted (1 skipped: too few OTM points); params in-range")

# --- D. results coherence + deliverables ---------------------------------------
assert dq["gate_decision"] == "PROCEED"
assert arb["butterfly"]["n_violations"] == 0 and arb["calendar"]["n_pairs_violated"] == 0
assert bfly["n_post_violations"] == 0
assert qc["all_interp_butterfly_ok"] and qc["all_interp_calendar_ok"]
assert qc["rmse_iv_median"] < 0.01 and arb["rmse_iv_median"] < 0.01
from pathlib import Path
deliv = ["results/data_quality.json", "results/arb_violations.json",
         "results/surface_qc.json", "results/svi_butterfly_log.json",
         "data/raw/manifest.json",
         "results/plots/forward_curves.png", "results/plots/svi_param_stability.png"]
deliv += [f"results/plots/surface_3d_{d['date']}.png" for d in qc["dates"]]
deliv += [f"results/plots/smile_vs_market_{d['date']}.png" for d in qc["dates"]]
missing = [d for d in deliv if not (Path(root) / d).exists()]
assert not missing, missing
print(f"D. results coherent (0 violations everywhere, gate PROCEED); "
      f"{len(deliv)} deliverable files all present")
print("FULL AUDIT: ALL CHECKS PASSED")
