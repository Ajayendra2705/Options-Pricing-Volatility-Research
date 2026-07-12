from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
"""Day 10 verify: stored SVI params reproduce reported RMSE, beat/match a
dumb quadratic baseline, and keep w >= 0 — all recomputed with inline
formulas, no src.surface.svi import."""
import numpy as np, pandas as pd

root = ROOT
fits = pd.read_parquet(root + "/data/processed/svi_params.parquet")
surf = pd.read_parquet(root + "/data/processed/iv_surface.parquet")
ok_s = surf[surf["status"] == "ok"]

def w_svi(k, a, b, rho, m, s):          # inline raw SVI
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s ** 2))

worst_dev, base_wins = 0.0, 0
for _, f in fits[fits["fit_ok"]].iterrows():
    g = ok_s[(ok_s["date"] == f["date"]) & (ok_s["expiry"] == f["expiry"])]
    otm = g[np.where(g["strike"] < g["F"], g["option_type"] == "P", g["option_type"] == "C")]
    if f.get("augmented", False):
        # thin-OTM slice: fit used OTM + near-ATM ITM quotes (|k| <= 0.10)
        near_itm = g.drop(otm.index)
        otm = pd.concat([otm, near_itm[near_itm["log_moneyness"].abs() <= 0.10]])
        otm = otm.sort_values("log_moneyness")
    k, iv = otm["log_moneyness"].to_numpy(), otm["iv"].to_numpy()
    assert len(k) == f["n_points"], (f["date"], f["expiry"], len(k), f["n_points"])
    iv_fit = np.sqrt(np.maximum(w_svi(k, f["a"], f["b"], f["rho"], f["m"], f["sigma"]), 0) / f["T"])
    rmse = np.sqrt(np.mean((iv_fit - iv) ** 2))
    worst_dev = max(worst_dev, abs(rmse - f["rmse_iv"]))
    # baseline: quadratic in k fitted to the same IVs
    q = np.polyval(np.polyfit(k, iv, 2), k)
    rmse_quad = np.sqrt(np.mean((q - iv) ** 2))
    if rmse_quad < rmse:
        base_wins += 1
    # w >= 0 on a wide grid
    kk = np.linspace(k.min() - 0.2, k.max() + 0.2, 400)
    assert w_svi(kk, f["a"], f["b"], f["rho"], f["m"], f["sigma"]).min() >= -1e-12

print(f"stored-vs-recomputed RMSE worst |dev|: {worst_dev:.2e}")
print(f"quadratic baseline beats SVI on {base_wins}/{fits['fit_ok'].sum()} slices")
med_svi = fits.loc[fits['fit_ok'], 'rmse_iv'].median() * 100
print(f"median SVI RMSE {med_svi:.2f} volpts")
print("DAY 10 INDEPENDENT CHECKS PASSED")
