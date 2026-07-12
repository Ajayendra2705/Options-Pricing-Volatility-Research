from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[2])
import pandas as pd, numpy as np
root = ROOT
fwd = pd.read_parquet(root + "/data/processed/forwards.parquet")
ch = pd.read_parquet(root + "/data/processed/chain_clean.parquet")

# 1. independent recompute: single most-ATM strike parity, no module code
errs = []
for _, row in fwd.iterrows():
    g = ch[(ch.date == row.date) & (ch.expiry == row.expiry)]
    piv = g.pivot_table(index="strike", columns="option_type", values="mid").dropna()
    cmp_ = piv.C - piv.P
    k_atm = cmp_.abs().idxmin()
    F_single = k_atm + cmp_[k_atm] / np.exp(-0.0525 * row["T"])
    errs.append(abs(F_single - row.F) / row.F)
print(f"single-ATM-strike vs parquet F: worst rel diff {max(errs):.4%} over {len(fwd)} slices")

# 2. external sanity: shortest-tenor F vs known AAPL closes (public record)
closes = {"2023-06-02": 180.95, "2023-06-07": 177.82, "2023-06-09": 180.96,
          "2023-06-12": 183.79, "2023-06-14": 183.95}
for d, grp in fwd.groupby("date"):
    row = grp.loc[grp["T"].idxmin()]
    s = closes[str(pd.Timestamp(d).date())]
    pred_bp = (0.0525 - 0.005) * row["T"] * 1e4
    print(f"{pd.Timestamp(d).date()}: F_short={row.F:.2f} close={s:.2f} "
          f"basis={(row.F / s - 1) * 1e4:+.0f}bp  carry-predicted ~+{pred_bp:.0f}bp (T={row['T']:.3f})")
