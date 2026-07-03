"""
Day 17 — HAR-RV forecast (Corsi 2009), log-vol specification.

Daily variance proxy (annualized), computable from one OHLC bar + prior close:

    v_t = 252 * ( ln(O_t/C_{t-1})^2 + RS_t )
    RS_t = ln(H/O)ln(H/C) + ln(L/O)ln(L/C)        (Rogers-Satchell, >= 0)

overnight^2 captures the gap; RS is the drift-independent intraday term.
HAR regressors are trailing vol aggregates of v (all information <= close of t):

    sig_d(t) = sqrt(v_t)
    sig_w(t) = sqrt(mean v_{t-4..t})               (weekly, 5 bars)
    sig_m(t) = sqrt(mean v_{t-21..t})              (monthly, 22 bars)

Model (log space, guarantees positive forecasts):

    ln sig_fwd(t) = b0 + bd ln sig_d(t) + bw ln sig_w(t) + bm ln sig_m(t) + e
    sig_fwd(t)    = sqrt(mean v_{t+1..t+h}),  h = HORIZON (default 21)

Point forecast uses the lognormal half-variance correction
exp(yhat + s^2/2) with s^2 = training residual variance, so the forecast
targets the mean (plain exp would target the median).

LOOKAHEAD CONVENTION: regressors at t use bars <= t. The expanding
out-of-sample forecast at t is fit only on rows i <= t - h (target fully
realized by close of t), then applied to x_t. In-sample fit/stats use the
full sample and are diagnostics only — the Day-18 signal must consume the
expanding column.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.realized_vol import RAW_DIR, TRADING_DAYS, _validate_ohlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

HORIZON = 21          # forecast target: mean variance over next 21 bars
WEEK, MONTH = 5, 22   # HAR aggregation windows
MIN_TRAIN = 100       # expanding forecast needs this many fitted rows
_VAR_FLOOR = 1e-12    # keeps ln() finite on a flat synthetic bar


def daily_variance_proxy(df: pd.DataFrame) -> pd.Series:
    """Per-day annualized variance: overnight^2 + Rogers-Satchell. Row 0 NaN."""
    df = _validate_ohlc(df)
    o = np.log(df["open"] / df["close"].shift(1))
    rs = (np.log(df["high"] / df["open"]) * np.log(df["high"] / df["close"])
          + np.log(df["low"] / df["open"]) * np.log(df["low"] / df["close"]))
    v = TRADING_DAYS * (o ** 2 + rs)
    v.index = pd.Index(df["date"], name="date")
    return v.rename("v_daily")


def har_dataset(df: pd.DataFrame, h: int = HORIZON) -> pd.DataFrame:
    """Regressors sig_d/w/m (trailing) and target sig_fwd (next h bars)."""
    v = daily_variance_proxy(df)
    out = pd.DataFrame({
        "sig_d": np.sqrt(np.maximum(v, _VAR_FLOOR)),
        "sig_w": np.sqrt(np.maximum(v.rolling(WEEK, min_periods=WEEK).mean(), _VAR_FLOOR)),
        "sig_m": np.sqrt(np.maximum(v.rolling(MONTH, min_periods=MONTH).mean(), _VAR_FLOOR)),
    }, index=v.index)
    # target: mean variance over t+1 .. t+h  (trailing mean shifted back)
    fwd = v.rolling(h, min_periods=h).mean().shift(-h)
    out["sig_fwd"] = np.sqrt(np.maximum(fwd, _VAR_FLOOR))
    return out


def _design(ds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X with intercept, y, row-valid mask) in log space."""
    X = np.column_stack([
        np.ones(len(ds)),
        np.log(ds["sig_d"].to_numpy()),
        np.log(ds["sig_w"].to_numpy()),
        np.log(ds["sig_m"].to_numpy()),
    ])
    y = np.log(ds["sig_fwd"].to_numpy())
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X, y, ok


def fit_har(ds: pd.DataFrame) -> dict:
    """OLS in log space on all complete rows. Returns coeffs + fit stats."""
    X, y, ok = _design(ds)
    if ok.sum() < 10:
        raise ValueError(f"HAR fit needs >= 10 complete rows, got {ok.sum()}")
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    resid = y[ok] - X[ok] @ beta
    s2 = resid.var(ddof=X.shape[1])
    r2 = 1.0 - resid.var(ddof=0) / y[ok].var(ddof=0)
    fitted = np.full(len(ds), np.nan)
    fitted[ok] = np.exp(X[ok] @ beta + 0.5 * s2)
    return {"beta": beta, "resid_var": s2, "r2": r2, "n": int(ok.sum()),
            "fitted": pd.Series(fitted, index=ds.index, name="har_insample")}


def expanding_forecast(ds: pd.DataFrame, h: int = HORIZON,
                       min_train: int = MIN_TRAIN) -> pd.Series:
    """No-lookahead HAR forecast: at t, fit on rows i <= t - h only."""
    X, y, ok = _design(ds)
    n = len(ds)
    out = np.full(n, np.nan)
    for t in range(n):
        if not (np.isfinite(X[t]).all()):
            continue
        train = ok.copy()
        train[max(t - h + 1, 0):] = False          # target must be realized by t
        if train.sum() < min_train:
            continue
        beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        resid = y[train] - X[train] @ beta
        out[t] = np.exp(X[t] @ beta + 0.5 * resid.var(ddof=X.shape[1]))
    return pd.Series(out, index=ds.index, name="har_oos")


def plot_har(tab: pd.DataFrame, out_dir: Path = PLOTS_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(tab["date"], tab["sig_fwd"] * 100, "-", lw=1.4, color="0.3",
            label=f"realized fwd {HORIZON}d vol")
    ax.plot(tab["date"], tab["har_insample"] * 100, "--", lw=1.2,
            label="HAR in-sample fit")
    ax.plot(tab["date"], tab["har_oos"] * 100, "-", lw=1.2,
            label="HAR expanding forecast (no lookahead)")
    ax.set_ylabel("annualized vol (%)")
    ax.legend()
    ax.set_title("HAR-RV: forecast vs realized")
    p = out_dir / "har_forecast.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def run_har(ohlc_path: Path | None = None) -> pd.DataFrame:
    """Day 17 deliverable: HAR forecast table + stats + plot."""
    df = pd.read_parquet(ohlc_path or RAW_DIR / "aapl_ohlc.parquet")
    ds = har_dataset(df)
    fit = fit_har(ds)
    ds["har_insample"] = fit["fitted"]
    ds["har_oos"] = expanding_forecast(ds)
    tab = ds.reset_index()

    out = PROCESSED_DIR / "har_forecast.parquet"
    tab.to_parquet(out, index=False)

    both = tab.dropna(subset=["har_oos", "sig_fwd"])
    oos = {
        "n": int(len(both)),
        "corr": float(np.corrcoef(both["har_oos"], both["sig_fwd"])[0, 1])
        if len(both) >= 3 else None,
        "rmse_volpts": float(np.sqrt(((both["har_oos"] - both["sig_fwd"]) ** 2).mean()) * 100)
        if len(both) else None,
    }
    stats = {
        "spec": "ln sig_fwd ~ 1 + ln sig_d + ln sig_w + ln sig_m (h=%d)" % HORIZON,
        "beta": {k: float(b) for k, b in
                 zip(("const", "daily", "weekly", "monthly"), fit["beta"])},
        "r2_insample": float(fit["r2"]),
        "resid_var": float(fit["resid_var"]),
        "n_insample": fit["n"],
        "oos_expanding": oos,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "har_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"HAR: n={fit['n']} in-sample R2={fit['r2']:.3f} "
          f"betas d/w/m = {fit['beta'][1]:.2f}/{fit['beta'][2]:.2f}/{fit['beta'][3]:.2f}")
    if oos["corr"] is not None:
        print(f"OOS expanding: n={oos['n']} corr={oos['corr']:.3f} "
              f"rmse={oos['rmse_volpts']:.2f} volpts")
    print(f"-> {out}")
    print(f"-> {RESULTS_DIR / 'har_stats.json'}")
    print(f"-> {plot_har(tab)}")
    return tab


if __name__ == "__main__":
    run_har()
