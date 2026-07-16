"""
Day 18 — Signal construction (pre-registered primary: config/primary.yaml).

Per (quote date, expiry) slice:

    signal_raw = atm_iv - har_forecast_h        [vol points]

- atm_iv: SVI total variance at k=0 from the joint arb-free fit,
  iv = sqrt(w(0)/T).
- har_forecast_h: HAR expanding out-of-sample forecast (no lookahead; see
  src/backtest/har.py) with horizon matched to the option's remaining life,
  h = clamp(round(252*T), H_MIN, H_MAX). Both legs use information available
  at the close of the quote date only.

Ranking (primary): within each quote date, rank slices by signal_raw
descending; rank 1 -> short_vol, last rank -> long_vol, middle -> flat.

signal_z (DIAGNOSTIC only, never trade selection in the primary): per tenor
bucket, (raw - trailing mean) / trailing std over strictly PRIOR quote dates,
NaN until MIN_Z_OBS prior observations exist. With 5 quote dates this is
noise — which is exactly why the primary config declares normalization: none.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.har import expanding_forecast, har_dataset
from src.backtest.realized_vol import RAW_DIR
from src.surface.svi import svi_total_variance

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

H_MIN, H_MAX = 5, 63
MIN_Z_OBS = 3
TENOR_EDGES = ((0.06, "short"), (0.10, "mid"), (np.inf, "long"))


def tenor_bucket(T: float) -> str:
    for edge, name in TENOR_EDGES:
        if T <= edge:
            return name
    raise AssertionError("unreachable")


def horizon(T: float) -> int:
    return int(np.clip(round(252 * T), H_MIN, H_MAX))


def atm_iv(row: pd.Series) -> float:
    w0 = float(svi_total_variance(0.0, np.array(
        [row["a"], row["b"], row["rho"], row["m"], row["sigma"]])))
    return float(np.sqrt(max(w0, 0.0) / row["T"]))


def har_forecasts_by_horizon(ohlc: pd.DataFrame,
                             horizons: set[int]) -> dict[int, pd.Series]:
    """Expanding OOS HAR forecast series (indexed by date), one per horizon."""
    return {h: expanding_forecast(har_dataset(ohlc, h), h) for h in sorted(horizons)}


def build_signal(params: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """Signal table per (date, expiry). Only fit_ok slices participate."""
    rows = params[params["fit_ok"]].copy()
    rows["atm_iv"] = rows.apply(atm_iv, axis=1)
    rows["bucket"] = rows["T"].map(tenor_bucket)
    rows["h"] = rows["T"].map(horizon)

    fcst = har_forecasts_by_horizon(ohlc, set(rows["h"]))
    rows["rv_fcst"] = [
        float(fcst[h].get(d, np.nan)) for d, h in zip(rows["date"], rows["h"])
    ]
    rows["signal_raw"] = rows["atm_iv"] - rows["rv_fcst"]

    # diagnostic z per tenor bucket: strictly prior quote dates only
    rows = rows.sort_values(["bucket", "date"]).reset_index(drop=True)
    g = rows.groupby("bucket")["signal_raw"]
    mu = g.transform(lambda s: s.expanding(MIN_Z_OBS).mean().shift(1))
    sd = g.transform(lambda s: s.expanding(MIN_Z_OBS).std(ddof=1).shift(1))
    rows["signal_z"] = (rows["signal_raw"] - mu) / sd

    # primary ranking: within date, raw signal descending; NaN signals excluded
    rows["rank"] = rows.groupby("date")["signal_raw"].rank(
        ascending=False, method="first")
    n_by_date = rows.groupby("date")["rank"].transform("max")
    rows["side"] = np.select(
        [rows["rank"].isna(), rows["rank"] == 1.0, rows["rank"] == n_by_date],
        ["flat", "short_vol", "long_vol"], default="flat")

    cols = ["date", "expiry", "T", "bucket", "h", "atm_iv", "rv_fcst",
            "signal_raw", "signal_z", "rank", "side"]
    return rows.sort_values(["date", "T"]).reset_index(drop=True)[cols]


def plot_signal(tab: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for bucket, mk in (("short", "o"), ("mid", "s"), ("long", "^")):
        b = tab[tab["bucket"] == bucket]
        ax.plot(b["date"], b["signal_raw"] * 100, mk + "-", label=bucket)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_ylabel("ATM IV − HAR forecast (vol pts)")
    ax.legend(title="tenor bucket")
    ax.set_title("Vol-arb signal by tenor bucket")
    p = out_dir / "signal.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def run_signal(
    ohlc_path: Path | None = None,
    processed_dir: Path | None = None,
    plots_dir: Path | None = None,
    summary_path: Path | None = None,
    make_plots: bool = True,
    config_label: str = "config/primary.yaml (pre-registered)",
) -> pd.DataFrame:
    """Day 18 deliverable: signal time-series + side assignment.

    Seams default to v1's constants (Day 32 convention) so v1's paths never
    move; `summary_path` keeps a Phase-2 run out of the tracked
    results/signal_summary.json, `config_label` names the pre-registration the
    run answers to.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    plots_dir = plots_dir or PLOTS_DIR
    summary_path = summary_path or RESULTS_DIR / "signal_summary.json"
    params = pd.read_parquet(processed_dir / "svi_params_joint.parquet")
    ohlc = pd.read_parquet(ohlc_path or RAW_DIR / "aapl_ohlc.parquet")
    tab = build_signal(params, ohlc)

    out = processed_dir / "signal.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out, index=False)

    ok = tab.dropna(subset=["signal_raw"])
    summary = {
        "config": config_label,
        "n_slices": int(len(tab)),
        "n_with_signal": int(len(ok)),
        "signal_volpts": {
            "mean": float(ok["signal_raw"].mean() * 100),
            "min": float(ok["signal_raw"].min() * 100),
            "max": float(ok["signal_raw"].max() * 100),
        },
        "sides": {k: int(v) for k, v in tab["side"].value_counts().items()},
        "by_bucket_mean_volpts": {
            k: float(v * 100)
            for k, v in ok.groupby("bucket")["signal_raw"].mean().items()},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="\n") as f:
        json.dump(summary, f, indent=2)

    print(f"signal: {len(ok)}/{len(tab)} slices, "
          f"mean {summary['signal_volpts']['mean']:.2f} volpts, "
          f"sides {summary['sides']}")
    print(f"-> {out}")
    print(f"-> {summary_path}")
    if make_plots:
        print(f"-> {plot_signal(tab, plots_dir)}")
    return tab


if __name__ == "__main__":
    run_signal()
