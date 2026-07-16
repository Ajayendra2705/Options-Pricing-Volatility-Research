"""
Day 7 — Forward construction from put-call parity.

For European options, C(K) - P(K) = df * (F - K) exactly, so
F_k = K + (C - P)/df at every strike. Method here (practitioner standard):

  1. Fix df = exp(-r*T) from an EXTERNAL risk-free rate (config below;
     June-2023 3M T-bill ~5.25%). Estimating df from the chain slope is
     ill-conditioned at short tenors (r = -ln(df)/T blows penny noise up
     by 1/T) and biased for American underlyings — kept only as a
     diagnostic (`r_implied`).
  2. Compute F_k per strike; take the median of the ATM_WINDOW strikes
     with smallest |C - P| (closest to the forward). Near ATM, C - P ~ 0,
     so an error in df barely moves F — and the American early-exercise
     premium is smallest there.
  3. Stability diagnostic across ALL strikes (PLAN.md test): F_std, F_range.
  4. Carry per date from the forward term structure: ln F is ~linear in T
     with slope (r - q), so q_implied = r - slope. No spot needed
     (the vendor chain has no underlying column — spot+r comparison is
     replaced by this term-structure carry sanity, documented).

American-parity caveat: deep-ITM put EEP inflates P at high strikes, which
is why the all-strike regression implies df > 1 (negative rates) on AAPL.
The ATM-window median sidesteps it; the wing bias is visible in F_range.

Outputs (run_forwards): data/processed/forwards.parquet + per-date curve
plot results/plots/forward_curves.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

RISK_FREE_RATE = 0.0525   # 3M T-bill, June 2023 window of the AAPL dataset
MIN_PAIRS = 3
ATM_WINDOW = 5            # strikes nearest the forward used for the F median


def imply_forward_slice(slice_df: pd.DataFrame, r: float = RISK_FREE_RATE) -> dict | None:
    """Imply the forward for one (date, expiry) slice.

    Needs columns: strike, option_type (C/P), mid, T.
    Returns None when fewer than MIN_PAIRS call/put strike pairs exist.
    """
    piv = slice_df.pivot_table(index="strike", columns="option_type", values="mid")
    if not {"C", "P"}.issubset(piv.columns):
        return None
    piv = piv.dropna()
    if len(piv) < MIN_PAIRS:
        return None

    T = float(slice_df["T"].iloc[0])
    df = float(np.exp(-r * T))
    K = piv.index.to_numpy(float)
    cmp_ = (piv["C"] - piv["P"]).to_numpy(float)

    F_k = K + cmp_ / df                                  # per-strike parity forwards
    atm = np.argsort(np.abs(cmp_))[: min(ATM_WINDOW, len(K))]
    F = float(np.median(F_k[atm]))

    # diagnostic only: chain-implied discount from the all-strike slope
    slope, _ = np.polyfit(K, cmp_, 1)
    r_implied = float(-np.log(-slope) / T) if slope < 0 else np.nan

    return {
        "F": F,
        "df": df,
        "r": r,
        "r_implied": r_implied,
        "T": T,
        "n_pairs": int(len(K)),
        "F_std": float(F_k.std(ddof=0)),
        "F_range": float(F_k.max() - F_k.min()),
        "F_std_atm": float(F_k[atm].std(ddof=0)),
    }


def imply_forwards(chain: pd.DataFrame, r: float = RISK_FREE_RATE) -> pd.DataFrame:
    """Parity forwards for every (date, expiry) slice in a cleaned chain."""
    rows = []
    for (date, expiry), grp in chain.groupby(["date", "expiry"]):
        res = imply_forward_slice(grp, r)
        if res is not None:
            rows.append({"date": date, "expiry": expiry, **res})
    out = pd.DataFrame(rows)
    return out.sort_values(["date", "expiry"]).reset_index(drop=True) if len(out) else out


def implied_carry(fwd: pd.DataFrame) -> pd.DataFrame:
    """Per quote date: q = r - slope of ln F vs T (forward term structure).

    Needs >= 2 expiries per date. Replaces the spot+r sanity check (no spot
    column in the vendor chain): carry must come out near the dividend yield.
    """
    rows = []
    for date, grp in fwd.groupby("date"):
        if len(grp) < 2:
            continue
        slope, _ = np.polyfit(grp["T"], np.log(grp["F"]), 1)
        rows.append({"date": date, "n_expiries": len(grp),
                     "carry_slope": float(slope),                # = r - q
                     "q_implied": float(grp["r"].iloc[0] - slope)})
    return pd.DataFrame(rows)


def plot_forward_curves(fwd: pd.DataFrame, out_path: Path) -> None:
    """One line per quote date: F vs expiry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for date, grp in fwd.groupby("date"):
        ax.plot(pd.to_datetime(grp["expiry"]), grp["F"], marker="o",
                label=str(pd.Timestamp(date).date()))
    ax.set_xlabel("Expiry")
    ax.set_ylabel("Parity-implied forward F")
    ax.set_title("Forward curves per quote date (put-call parity, ATM window)")
    ax.legend(title="Quote date", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_forwards(
    chain_path: Path | None = None,
    out_path: Path | None = None,
    plot_path: Path | None = None,
    r: float = RISK_FREE_RATE,
    processed_dir: Path | None = None,
    plots_dir: Path | None = None,
    make_plots: bool = True,
) -> pd.DataFrame:
    """Load cleaned chain -> forwards table + carry + curve plot.

    `processed_dir`/`plots_dir` redirect the whole stage (Phase 2 runs the same
    code on SPY into data/phase2/**); explicit *_path args still win. Defaults
    are v1's constants, so v1's paths cannot move.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    plots_dir = plots_dir or PLOTS_DIR
    chain_path = chain_path or processed_dir / "chain_clean.parquet"
    chain = pd.read_parquet(chain_path)
    fwd = imply_forwards(chain, r)
    if fwd.empty:
        raise RuntimeError("No slice produced a parity forward — check the cleaned chain.")

    out_path = out_path or processed_dir / "forwards.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fwd.to_parquet(out_path, index=False)
    if make_plots:
        plot_forward_curves(fwd, plot_path or plots_dir / "forward_curves.png")

    carry = implied_carry(fwd)
    rel_stab = (fwd["F_std_atm"] / fwd["F"]).max()
    print(
        f"forwards: {len(fwd)} slices | F range [{fwd['F'].min():.2f}, {fwd['F'].max():.2f}] | "
        f"worst ATM F std/F {rel_stab:.4%}"
    )
    if len(carry):
        print(f"implied carry per date (q = r - dlnF/dT): "
              f"{', '.join(f'{pd.Timestamp(d).date()}: {q:+.2%}' for d, q in zip(carry['date'], carry['q_implied']))}")
    print(f"-> {out_path}")
    return fwd


if __name__ == "__main__":
    run_forwards()
