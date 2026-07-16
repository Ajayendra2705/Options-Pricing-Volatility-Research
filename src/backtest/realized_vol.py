"""
Day 16 — Realized volatility estimators (trailing, no lookahead).

Yang-Zhang (2000): minimum-variance unbiased combination of overnight,
open-to-close and Rogers-Satchell variances, drift-independent:

    sigma^2_YZ = sigma^2_ON + k * sigma^2_OC + (1 - k) * sigma^2_RS
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

with, over a window of n bars (sample variances, ddof=1):
    overnight  o_t = ln(O_t / C_{t-1})
    open-close c_t = ln(C_t / O_t)
    RS term    rs_t = ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)      (mean, not var)

Close-to-close kept as the baseline cross-check.

LOOKAHEAD CONVENTION: the estimate indexed at date t uses bars
(t - n + 1) .. t inclusive — information available at the close of t, never
after. The first (n) rows are NaN (overnight return needs C_{t-1}).
Everything is trailing `rolling(...)`; no centering, no future bars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

TRADING_DAYS = 252
DEFAULT_WINDOW = 21


def _validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("date", "open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"OHLC frame missing column '{col}'")
    out = df.sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise ValueError("duplicate dates in OHLC frame")
    return out


def close_to_close_vol(df: pd.DataFrame, window: int = DEFAULT_WINDOW,
                       annualize: bool = True) -> pd.Series:
    """Trailing close-to-close vol: std of ln(C_t/C_{t-1}) over the window."""
    df = _validate_ohlc(df)
    r = np.log(df["close"] / df["close"].shift(1))
    var = r.rolling(window, min_periods=window).var(ddof=1)
    scale = TRADING_DAYS if annualize else 1.0
    out = np.sqrt(var * scale)
    out.index = pd.Index(df["date"], name="date")
    return out.rename("cc_vol")


def yang_zhang_vol(df: pd.DataFrame, window: int = DEFAULT_WINDOW,
                   annualize: bool = True) -> pd.Series:
    """Trailing Yang-Zhang vol. Estimate at t uses bars t-window+1 .. t."""
    df = _validate_ohlc(df)
    o = np.log(df["open"] / df["close"].shift(1))      # overnight
    c = np.log(df["close"] / df["open"])               # open-to-close
    rs = (np.log(df["high"] / df["open"]) * np.log(df["high"] / df["close"])
          + np.log(df["low"] / df["open"]) * np.log(df["low"] / df["close"]))

    n = window
    var_on = o.rolling(n, min_periods=n).var(ddof=1)
    var_oc = c.rolling(n, min_periods=n).var(ddof=1)
    mean_rs = rs.rolling(n, min_periods=n).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var = var_on + k * var_oc + (1.0 - k) * mean_rs
    scale = TRADING_DAYS if annualize else 1.0
    out = np.sqrt(np.maximum(var, 0.0) * scale)
    out.index = pd.Index(df["date"], name="date")
    return out.rename("yz_vol")


def realized_vol_table(df: pd.DataFrame,
                       windows: tuple[int, ...] = (10, 21, 63)) -> pd.DataFrame:
    """YZ + close-to-close vols for several trailing windows, one row per date."""
    df = _validate_ohlc(df)
    out = pd.DataFrame({"date": df["date"]}).set_index("date")
    for w in windows:
        out[f"yz_{w}"] = yang_zhang_vol(df, w).to_numpy()
        out[f"cc_{w}"] = close_to_close_vol(df, w).to_numpy()
    return out.reset_index()


def plot_realized_vol(tab: pd.DataFrame, out_dir: Path = PLOTS_DIR,
                      ticker: str = "AAPL") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col, style in (("yz_21", "-"), ("cc_21", "--"), ("yz_10", ":")):
        ax.plot(tab["date"], tab[col] * 100, style, lw=1.4, label=col)
    ax.set_ylabel("annualized vol (%)")
    ax.legend()
    ax.set_title(f"{ticker} trailing realized vol (Yang-Zhang vs close-to-close)")
    p = out_dir / "realized_vol.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def run_realized_vol(
    ohlc_path: Path | None = None,
    processed_dir: Path | None = None,
    plots_dir: Path | None = None,
    make_plots: bool = True,
    ticker: str = "AAPL",
) -> pd.DataFrame:
    """Day 16 deliverable: trailing RV table + plot.

    Seams default to v1's constants (Day 32 convention) so v1's paths never move.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    plots_dir = plots_dir or PLOTS_DIR
    df = pd.read_parquet(ohlc_path or RAW_DIR / "aapl_ohlc.parquet")
    tab = realized_vol_table(df)
    out = processed_dir / "realized_vol.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out, index=False)

    last = tab.dropna().iloc[-1]
    print(f"realized vol: {len(tab)} dates {tab['date'].min().date()} .. "
          f"{tab['date'].max().date()}")
    print(f"latest ({last['date'].date()}): YZ21 {last['yz_21']:.1%} | "
          f"CC21 {last['cc_21']:.1%} | YZ10 {last['yz_10']:.1%}")
    print(f"-> {out}")
    if make_plots:
        print(f"-> {plot_realized_vol(tab, plots_dir, ticker)}")
    return tab


if __name__ == "__main__":
    run_realized_vol()
