"""
Day 8 — Raw IV surface from the real cleaned chain.

Inverts every cleaned quote to a Black-76 implied vol using the Day-7
parity forwards (per date/expiry F, df) — the whole chain shares one
consistent forward per slice, so call and put IVs at the same strike must
agree up to microstructure noise (parity), a built-in consistency check.

Every row keeps a status:
    ok               finite IV
    below_intrinsic  mid under the discounted intrinsic (American bias /
                     stale wing) -> nan, flagged not dropped
    above_upper      mid above the sigma->inf bound -> nan
    no_solution      inside bounds but IV > 5.0 cap -> nan

DOCUMENTED BIAS — liquidity-conditioned wing selection: the Day-6 cleaning
drops zero-bid and wide-spread quotes, which live disproportionately in the
far wings. Surviving wing quotes are the *liquid* ones, so the observed
wing IVs are conditioned on liquidity (deep-OTM skew is measured only where
dealers still quote two-sided). Any wing conclusion downstream (SVI fit
quality, tail pricing) inherits this selection. Quantified per slice in the
report as wing coverage: quoted strike range / (2 * 4 * ATM sigma * sqrt(T)).

Outputs (run_iv_surface): data/processed/iv_surface.parquet + one scatter
panel per quote date results/plots/iv_scatter_<date>.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.greeks import black_scholes as bs
from src.greeks.iv_invert import SIGMA_HI, implied_vol

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def build_iv_surface(chain: pd.DataFrame, forwards: pd.DataFrame) -> pd.DataFrame:
    """Merge chain with per-slice forwards, invert every mid to IV.

    Returns the chain with added columns: F, df, r, iv, log_moneyness
    (ln(K/F)), status. Rows without a Day-7 forward are dropped (reported
    by caller via the row-count delta).
    """
    fw = forwards[["date", "expiry", "F", "df", "r"]]
    df = chain.merge(fw, on=["date", "expiry"], how="inner").copy()

    cp = np.where(df["option_type"] == "C", 1, -1)
    iv = implied_vol(df["mid"].to_numpy(float), df["F"].to_numpy(float),
                     df["strike"].to_numpy(float), df["T"].to_numpy(float),
                     df["r"].to_numpy(float), cp)
    df["iv"] = iv
    df["log_moneyness"] = np.log(df["strike"] / df["F"])

    # classify failures
    p_fwd = df["mid"] / df["df"]                       # undiscounted premium
    intrinsic = np.maximum(cp * (df["F"] - df["strike"]), 0.0)
    upper = np.where(cp > 0, df["F"], df["strike"])
    status = np.full(len(df), "ok", dtype=object)
    bad = ~np.isfinite(iv)
    status[bad & (p_fwd < intrinsic).to_numpy()] = "below_intrinsic"
    status[bad & (p_fwd > upper).to_numpy()] = "above_upper"
    status[bad & (status == "ok")] = "no_solution"     # in-bounds but > SIGMA_HI cap
    # zero-vol results are parity-degenerate wings, flag them too
    status[(iv == 0.0)] = "below_intrinsic"
    df["status"] = status
    return df


def surface_report(surf: pd.DataFrame) -> dict:
    """Success/failure counts + per-slice ATM vol and wing coverage."""
    counts = surf["status"].value_counts().to_dict()
    ok = surf[surf["status"] == "ok"]
    slices = []
    for (date, expiry), g in ok.groupby(["date", "expiry"]):
        atm = g.loc[g["log_moneyness"].abs().idxmin()]
        width = 4.0 * atm["iv"] * np.sqrt(atm["T"])    # +-4 sigma total-vol band
        coverage = (g["log_moneyness"].max() - g["log_moneyness"].min()) / (2 * width) if width > 0 else np.nan
        slices.append({"date": str(pd.Timestamp(date).date()), "expiry": str(pd.Timestamp(expiry).date()),
                       "atm_iv": float(atm["iv"]), "n_ok": len(g),
                       "wing_coverage_4sig": round(float(coverage), 3)})
    return {
        "n_rows": len(surf),
        "status_counts": counts,
        "success_rate": round(counts.get("ok", 0) / len(surf), 4) if len(surf) else 0.0,
        "slices": slices,
    }


def plot_iv_scatter(surf: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Per quote date: IV vs strike, one subplot per expiry, C/P separated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = surf[surf["status"] == "ok"]
    paths = []
    for date, g in ok.groupby("date"):
        expiries = sorted(g["expiry"].unique())
        ncols = min(3, len(expiries))
        nrows = -(-len(expiries) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                                 squeeze=False, sharey=True)
        for ax, expiry in zip(axes.flat, expiries):
            sl = g[g["expiry"] == expiry]
            for otype, marker, color in (("C", "o", "tab:blue"), ("P", "^", "tab:red")):
                s = sl[sl["option_type"] == otype]
                ax.scatter(s["strike"], s["iv"], s=18, marker=marker, alpha=0.75,
                           color=color, label=otype)
            ax.axvline(sl["F"].iloc[0], color="gray", ls="--", lw=1, label="F")
            ax.set_title(f"exp {pd.Timestamp(expiry).date()} (T={sl['T'].iloc[0]:.3f})", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        for ax in axes.flat[len(expiries):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Strike")
        for row_axes in axes:
            row_axes[0].set_ylabel("IV")
        d = pd.Timestamp(date).date()
        fig.suptitle(f"Raw IV scatter — {d}", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"iv_scatter_{d}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def run_iv_surface(
    chain_path: Path | None = None,
    forwards_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    chain = pd.read_parquet(chain_path or PROCESSED_DIR / "chain_clean.parquet")
    forwards = pd.read_parquet(forwards_path or PROCESSED_DIR / "forwards.parquet")

    surf = build_iv_surface(chain, forwards)
    report = surface_report(surf)

    out_path = out_path or PROCESSED_DIR / "iv_surface.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    surf.to_parquet(out_path, index=False)
    paths = plot_iv_scatter(surf, PLOTS_DIR)

    print(f"iv_surface: {report['n_rows']} quotes ({len(chain)} cleaned, "
          f"{len(chain) - report['n_rows']} without Day-7 forward) | "
          f"success {report['success_rate']:.1%} | status {report['status_counts']}")
    ok = surf[surf["status"] == "ok"]
    print(f"IV range [{ok['iv'].min():.3f}, {ok['iv'].max():.3f}] | "
          f"median wing coverage (of +-4sig) "
          f"{np.median([s['wing_coverage_4sig'] for s in report['slices']]):.2f}")
    print(f"-> {out_path}")
    print(f"-> {len(paths)} scatter panels in {PLOTS_DIR}")
    return surf


if __name__ == "__main__":
    run_iv_surface()
