"""
Day 9 — Raw SVI calibration (single slice).

Raw SVI (Gatheral) total variance in log-moneyness k = ln(K/F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

params (a, b, rho, m, sigma): b >= 0, |rho| < 1, sigma > 0, and
a + b*sigma*sqrt(1-rho^2) >= 0 keeps w >= 0 everywhere.

Fit: least squares in TOTAL VARIANCE space (w = iv^2 * T) with
scipy.optimize.least_squares (bounded, trf) from a small multi-start grid
over (m, sigma) — raw SVI's loss is multi-modal in those two.

Slice input: OTM side only (puts below F, calls above) — the liquid,
EEP-clean side of an American single-name chain; the ITM side's
early-exercise premium would drag the fit (Day 7/8 finding).

No-arb (butterfly/calendar) is NOT enforced here — that's Days 11-13.
Deliverable runner: fit one real slice, plot fit vs market, print RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])


def svi_total_variance(k, params: SVIParams | np.ndarray):
    """Raw SVI w(k)."""
    a, b, rho, m, sigma = (params.as_array() if isinstance(params, SVIParams) else np.asarray(params, float))
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_iv(k, T, params):
    """Implied vol from SVI total variance (clipped at 0 before sqrt)."""
    w = np.maximum(svi_total_variance(k, params), 0.0)
    return np.sqrt(w / T)


def otm_side(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Keep OTM quotes only: puts with K < F, calls with K >= F; status ok."""
    ok = slice_df[slice_df["status"] == "ok"] if "status" in slice_df.columns else slice_df
    is_otm = np.where(ok["strike"] < ok["F"], ok["option_type"] == "P", ok["option_type"] == "C")
    return ok[is_otm]


def fit_svi_slice(k, iv, T, weights=None) -> tuple[SVIParams, dict]:
    """Least-squares raw-SVI fit of one maturity slice.

    k: log-moneyness array, iv: implied vols, T: year fraction.
    weights: optional per-point weights on the total-variance residual.
    Returns (params, report) with report holding rmse_iv, rmse_w, n_points.
    """
    k = np.asarray(k, float)
    w_mkt = np.asarray(iv, float) ** 2 * T
    wt = np.ones_like(k) if weights is None else np.asarray(weights, float)

    span = max(k.max() - k.min(), 1e-3)
    w_max, w_min = w_mkt.max(), max(w_mkt.min(), 1e-8)

    def resid(p):
        return (svi_total_variance(k, p) - w_mkt) * wt

    # bounds: a can go slightly negative (w>=0 enforced post-hoc via check),
    # b positive, |rho|<1, m inside a widened k-range, sigma positive
    lb = [-w_max, 1e-8, -0.999, k.min() - span, 1e-4]
    ub = [w_max * 2 + 1e-8, 10.0 * (w_max / span + 1.0), 0.999, k.max() + span, 4.0 * span + 1.0]

    best, best_cost = None, np.inf
    for m0 in np.quantile(k, [0.25, 0.5, 0.75]):
        for s0 in (0.1 * span, 0.5 * span):
            p0 = np.clip([w_min, (w_max - w_min) / span + 1e-3, -0.5, m0, s0], lb, ub)
            try:
                sol = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=2000)
            except ValueError:
                continue
            if sol.cost < best_cost:
                best, best_cost = sol, sol.cost

    if best is None:
        raise RuntimeError("SVI fit failed from every start")

    params = SVIParams(*best.x)
    w_fit = svi_total_variance(k, params)
    iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T)
    report = {
        "rmse_iv": float(np.sqrt(np.mean((iv_fit - np.asarray(iv, float)) ** 2))),
        "rmse_w": float(np.sqrt(np.mean((w_fit - w_mkt) ** 2))),
        "n_points": int(len(k)),
        "min_w_on_grid": float(w_fit.min()),
        "cost": float(best.cost),
    }
    return params, report


MIN_POINTS = 6


def fit_all_slices(surf: pd.DataFrame) -> pd.DataFrame:
    """Day 10: raw-SVI fit for every (date, expiry) slice of an IV surface.

    Uses OTM side only. Slices with < MIN_POINTS quotes are skipped and
    reported with fit_ok = False. Returns one row per slice with params,
    RMSE and diagnostics.
    """
    rows = []
    for (date, expiry), g in surf.groupby(["date", "expiry"]):
        sl = otm_side(g)
        row = {"date": date, "expiry": expiry, "n_points": len(sl)}
        if len(sl) < MIN_POINTS:
            rows.append({**row, "fit_ok": False})
            continue
        T = float(sl["T"].iloc[0])
        try:
            params, report = fit_svi_slice(sl["log_moneyness"], sl["iv"], T)
        except RuntimeError:
            rows.append({**row, "fit_ok": False, "T": T})
            continue
        rows.append({
            **row, "fit_ok": True, "T": T,
            "a": params.a, "b": params.b, "rho": params.rho,
            "m": params.m, "sigma": params.sigma,
            "rmse_iv": report["rmse_iv"], "min_w_on_grid": report["min_w_on_grid"],
        })
    return pd.DataFrame(rows).sort_values(["date", "expiry"]).reset_index(drop=True)


def plot_param_stability(fits: pd.DataFrame, out_path: Path) -> None:
    """Param time-series across quote dates, one series per expiry.

    Smooth series across dates = stable calibration; jumps = overfit proxy
    (params trading off against each other on similar smiles).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = fits[fits["fit_ok"]].copy()
    ok["date"] = pd.to_datetime(ok["date"])
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    for ax, param in zip(axes, ["a", "b", "rho", "m", "sigma"]):
        for expiry, g in ok.groupby("expiry"):
            ax.plot(g["date"], g[param], marker="o",
                    label=str(pd.Timestamp(expiry).date()))
        ax.set_ylabel(param)
        ax.grid(alpha=0.3)
    axes[0].legend(title="Expiry", fontsize=7, ncol=3)
    axes[0].set_title("Raw-SVI param time-series per expiry (smoothness = overfit proxy)")
    axes[-1].set_xlabel("Quote date")
    fig.autofmt_xdate()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_svi_all(surface_path: Path | None = None) -> pd.DataFrame:
    """Day 10 deliverable: all-slice fit, RMSE table, param-stability plot."""
    surf = pd.read_parquet(surface_path or PROCESSED_DIR / "iv_surface.parquet")
    fits = fit_all_slices(surf)

    out_path = PROCESSED_DIR / "svi_params.parquet"
    fits.to_parquet(out_path, index=False)
    plot_param_stability(fits, PLOTS_DIR / "svi_param_stability.png")

    ok = fits[fits["fit_ok"]]
    print(f"SVI all-slice fit: {len(ok)}/{len(fits)} slices fitted")
    print("\nRMSE table (vol pts):")
    tab = ok.assign(
        date=lambda d: pd.to_datetime(d["date"]).dt.date,
        expiry=lambda d: pd.to_datetime(d["expiry"]).dt.date,
        rmse_volpts=lambda d: (d["rmse_iv"] * 100).round(2),
    )[["date", "expiry", "T", "n_points", "rmse_volpts", "rho", "min_w_on_grid"]]
    print(tab.to_string(index=False))
    print(f"\nmedian RMSE {ok['rmse_iv'].median() * 100:.2f} volpts | "
          f"max {ok['rmse_iv'].max() * 100:.2f} | "
          f"negative-w slices: {(ok['min_w_on_grid'] < 0).sum()}")
    print(f"-> {out_path}")
    print(f"-> {PLOTS_DIR / 'svi_param_stability.png'}")
    return fits


def fit_one_real_slice(surface_path: Path | None = None, plot: bool = True) -> tuple[SVIParams, dict]:
    """Day 9 deliverable: fit the most-quoted real slice, plot, print RMSE."""
    surf = pd.read_parquet(surface_path or PROCESSED_DIR / "iv_surface.parquet")
    ok = surf[surf["status"] == "ok"]
    date, expiry = ok.groupby(["date", "expiry"]).size().idxmax()
    sl = otm_side(ok[(ok["date"] == date) & (ok["expiry"] == expiry)])
    T = float(sl["T"].iloc[0])

    params, report = fit_svi_slice(sl["log_moneyness"], sl["iv"], T)
    d, e = pd.Timestamp(date).date(), pd.Timestamp(expiry).date()
    print(f"SVI fit {d} -> exp {e} (T={T:.3f}, n={report['n_points']} OTM quotes)")
    print(f"  params: a={params.a:.5f} b={params.b:.4f} rho={params.rho:+.3f} "
          f"m={params.m:+.4f} sigma={params.sigma:.4f}")
    print(f"  RMSE: {report['rmse_iv'] * 100:.2f} vol pts (w-space {report['rmse_w']:.6f})")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kk = np.linspace(sl["log_moneyness"].min() - 0.02, sl["log_moneyness"].max() + 0.02, 300)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sl["log_moneyness"], sl["iv"], s=25, color="tab:blue", label="market (OTM mids)")
        ax.plot(kk, svi_iv(kk, T, params), color="tab:red", lw=2, label="raw SVI fit")
        ax.set_xlabel("log-moneyness k = ln(K/F)")
        ax.set_ylabel("IV")
        ax.set_title(f"Raw SVI, {d} -> {e} | RMSE {report['rmse_iv'] * 100:.2f} vol pts")
        ax.grid(alpha=0.3)
        ax.legend()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        p = PLOTS_DIR / f"svi_fit_{d}_{e}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {p}")
    return params, report


if __name__ == "__main__":
    import sys

    if "--all" in sys.argv:
        run_svi_all()
    else:
        fit_one_real_slice()
