"""
Day 14 — Surface assembly + QC.

Stitch the arb-free joint SVI slices (Day 13) into a queryable vol surface
per quote date:

- In T, between expiry nodes: LINEAR interpolation of total variance w at
  fixed forward moneyness k = ln(K/F_T). Linear-in-w of calendar-ordered
  nodes is automatically calendar-monotone; no-butterfly of interpolated
  slices is NOT guaranteed by theory, so QC re-checks Durrleman g
  numerically (FD) on a dense T grid.
- Outside the node range: flat-IV extrapolation, w(k,T) = w(k,T_edge)*T/T_edge
  (keeps implied vol constant, preserves calendar monotonicity in both
  directions since w scales with T).
- Forwards: ln F linear in T between the Day-7 implied forwards (constant
  carry between nodes); edge-pair slope extrapolated outside.

QC report (results/surface_qc.json): per-date fit residuals vs market OTM
quotes, interpolated-slice butterfly minima, calendar monotonicity on a
dense T grid, coverage. Deliverables: 3D surface plot + smile-vs-market
panel per quote date, QC json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.surface.svi import SVIParams, otm_side, svi_iv, svi_total_variance

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

K_QC_SPAN = 1.0        # QC/plot range in log-moneyness (data lives within ~±0.5)
N_T_QC = 21            # interpolated T slices per date in the QC scan


@dataclass
class VolSurface:
    """Arb-free vol surface for one quote date: SVI slices + forward curve.

    The forward curve is decoupled from the vol nodes: an expiry whose vol
    slice failed to fit (too few OTM quotes) still carries a valid implied
    forward, so F_Ts/F_nodes may hold MORE nodes than Ts/params.
    """
    date: pd.Timestamp
    Ts: np.ndarray                 # vol node expiry times, sorted ascending
    params: list                   # SVIParams per vol node
    F_nodes: np.ndarray            # implied forwards, sorted by F_Ts
    expiries: list = field(default_factory=list)
    F_Ts: np.ndarray | None = None  # forward node times; defaults to Ts

    def w(self, k, T: float):
        """Total variance at forward moneyness k, time T (interp/extrap in T)."""
        k = np.asarray(k, float)
        Ts = self.Ts
        if T <= Ts[0]:
            return svi_total_variance(k, self.params[0]) * (T / Ts[0])
        if T >= Ts[-1]:
            return svi_total_variance(k, self.params[-1]) * (T / Ts[-1])
        i = int(np.searchsorted(Ts, T, side="right") - 1)
        lam = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
        return ((1.0 - lam) * svi_total_variance(k, self.params[i])
                + lam * svi_total_variance(k, self.params[i + 1]))

    def iv(self, k, T: float):
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / T)

    def forward(self, T: float) -> float:
        """ln F linear in T between nodes; edge slope extrapolated outside."""
        Ts = self.Ts if self.F_Ts is None else self.F_Ts
        lnF = np.log(self.F_nodes)
        if len(Ts) == 1:
            return float(self.F_nodes[0])
        if T <= Ts[0]:
            i = 0
        elif T >= Ts[-1]:
            i = len(Ts) - 2
        else:
            i = int(np.searchsorted(Ts, T, side="right") - 1)
        slope = (lnF[i + 1] - lnF[i]) / (Ts[i + 1] - Ts[i])
        return float(np.exp(lnF[i] + slope * (T - Ts[i])))

    def iv_strike(self, K, T: float):
        """Implied vol by cash strike (converts through the forward curve)."""
        return self.iv(np.log(np.asarray(K, float) / self.forward(T)), T)


def build_surfaces(fits: pd.DataFrame, forwards: pd.DataFrame) -> dict:
    """One VolSurface per quote date from joint fits + implied forwards."""
    surfaces = {}
    ok = fits[fits["fit_ok"] == True]  # noqa: E712 (object dtype)
    for date, g in ok.groupby("date"):
        g = g.sort_values("T")
        fw = forwards[forwards["date"] == date].sort_values("T")
        surfaces[date] = VolSurface(
            date=date,
            Ts=g["T"].to_numpy(float),
            params=[SVIParams(f.a, f.b, f.rho, f.m, f.sigma) for f in g.itertuples()],
            F_nodes=fw["F"].to_numpy(float),
            expiries=list(g["expiry"]),
            F_Ts=fw["T"].to_numpy(float),
        )
    return surfaces


def _g_fd(w_fun, k, h: float = 1e-5):
    """Durrleman g via finite differences on an arbitrary w(k) callable."""
    w = w_fun(k)
    wp = (w_fun(k + h) - w_fun(k - h)) / (2 * h)
    wpp = (w_fun(k + h) - 2 * w + w_fun(k - h)) / h**2
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1 - k * wp / (2 * w)) ** 2 - (wp**2 / 4) * (1 / w + 0.25) + wpp / 2
    return np.where(w > 0, g, -np.inf)


def qc_surface(vs: VolSurface, market: pd.DataFrame,
               k_span: float = K_QC_SPAN, n_t: int = N_T_QC) -> dict:
    """QC one date: market residuals + arb checks on interpolated slices."""
    kg = np.linspace(-k_span, k_span, 801)

    # fit-vs-market residuals on the OTM quotes the fits were built from
    resid = []
    for expiry, g in market[market["date"] == vs.date].groupby("expiry"):
        sl = otm_side(g)
        if not len(sl):
            continue
        T = float(sl["T"].iloc[0])
        err = vs.iv(sl["log_moneyness"].to_numpy(), T) - sl["iv"].to_numpy()
        resid.append(pd.Series(err, name=expiry))
    all_err = np.concatenate([r.to_numpy() for r in resid]) if resid else np.array([np.nan])

    # arb checks across interpolated/extrapolated T slices
    t_grid = np.linspace(0.5 * vs.Ts[0], 1.25 * vs.Ts[-1], n_t)
    min_g = min(float(_g_fd(lambda kk, t=t: vs.w(kk, t), kg).min()) for t in t_grid)
    w_stack = np.array([vs.w(kg, t) for t in t_grid])
    max_cal_decrease = float(np.maximum(w_stack[:-1] - w_stack[1:], 0.0).max())

    return {
        "date": str(pd.Timestamp(vs.date).date()),
        "n_expiries": len(vs.Ts),
        "T_range": [float(vs.Ts[0]), float(vs.Ts[-1])],
        "n_market_quotes": int(all_err.size),
        "rmse_iv": float(np.sqrt(np.mean(all_err**2))),
        "max_abs_err_iv": float(np.abs(all_err).max()),
        "frac_within_1volpt": float((np.abs(all_err) < 0.01).mean()),
        "interp_min_g": min_g,
        "interp_butterfly_ok": bool(min_g >= 0),
        "interp_max_calendar_decrease": max_cal_decrease,
        "interp_calendar_ok": bool(max_cal_decrease <= 1e-12),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_surface_3d(vs: VolSurface, out_dir: Path = PLOTS_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    kk = np.linspace(-0.35, 0.35, 81)
    tt = np.linspace(vs.Ts[0], vs.Ts[-1], 60)
    KK, TT = np.meshgrid(kk, tt)
    IV = np.array([vs.iv(kk, t) for t in tt]) * 100

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(KK, TT, IV, cmap="viridis", rstride=1, cstride=2,
                    linewidth=0, antialiased=True, alpha=0.95)
    for T, p in zip(vs.Ts, vs.params):                    # node slices overlaid
        ax.plot(kk, np.full_like(kk, T), svi_iv(kk, T, p) * 100, "k-", lw=1.2)
    ax.set_xlabel("log-moneyness k = ln(K/F)")
    ax.set_ylabel("T (years)")
    ax.set_zlabel("implied vol (%)")
    d = pd.Timestamp(vs.date).date()
    ax.set_title(f"AAPL SVI surface — {d}")
    ax.view_init(elev=22, azim=-60)
    p = out_dir / f"surface_3d_{d}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_smiles_vs_market(vs: VolSurface, market: pd.DataFrame,
                          out_dir: Path = PLOTS_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    day = market[market["date"] == vs.date]
    groups = sorted(day.groupby("expiry"), key=lambda kv: kv[1]["T"].iloc[0])
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 3.6),
                             squeeze=False, sharey=True)
    for ax, (expiry, g) in zip(axes[0], groups):
        sl = otm_side(g)
        T = float(g["T"].iloc[0])
        kk = np.linspace(g["log_moneyness"].min() - 0.02,
                         g["log_moneyness"].max() + 0.02, 200)
        ax.plot(kk, vs.iv(kk, T) * 100, "-", color="tab:blue", lw=1.6,
                label="SVI (arb-free)", zorder=1)
        other = g[(g["status"] == "ok") & ~g.index.isin(sl.index)]
        ax.plot(other["log_moneyness"], other["iv"] * 100, ".", color="0.75",
                ms=5, label="market (ITM side, unused)", zorder=2)
        ax.plot(sl["log_moneyness"], sl["iv"] * 100, "o", color="tab:red",
                ms=4, label="market OTM (fit input)", zorder=3)
        ax.axvline(0.0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(f"{pd.Timestamp(expiry).date()}  (T={T:.3f})", fontsize=10)
        ax.set_xlabel("k = ln(K/F)")
    axes[0][0].set_ylabel("implied vol (%)")
    axes[0][0].legend(fontsize=8)
    d = pd.Timestamp(vs.date).date()
    fig.suptitle(f"SVI smile vs market — {d}", y=1.02)
    p = out_dir / f"smile_vs_market_{d}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_assembly(fits_path: Path | None = None) -> dict:
    """Day 14 deliverable: surfaces + QC json + 3D/smile plots."""
    fits = pd.read_parquet(fits_path or PROCESSED_DIR / "svi_params_joint.parquet")
    forwards = pd.read_parquet(PROCESSED_DIR / "forwards.parquet")
    market = pd.read_parquet(PROCESSED_DIR / "iv_surface.parquet")

    surfaces = build_surfaces(fits, forwards)
    per_date = [qc_surface(vs, market) for vs in surfaces.values()]

    report = {
        "n_dates": len(per_date),
        "n_slices_total": int(sum(d["n_expiries"] for d in per_date)),
        "all_interp_butterfly_ok": bool(all(d["interp_butterfly_ok"] for d in per_date)),
        "all_interp_calendar_ok": bool(all(d["interp_calendar_ok"] for d in per_date)),
        "worst_interp_min_g": float(min(d["interp_min_g"] for d in per_date)),
        "rmse_iv_median": float(np.median([d["rmse_iv"] for d in per_date])),
        "max_abs_err_iv": float(max(d["max_abs_err_iv"] for d in per_date)),
        "frac_within_1volpt": float(np.mean([d["frac_within_1volpt"] for d in per_date])),
        "dates": per_date,
    }
    qc_path = PROJECT_ROOT / "results" / "surface_qc.json"
    qc_path.write_text(json.dumps(report, indent=2))

    n_plots = 0
    for vs in surfaces.values():
        plot_surface_3d(vs)
        plot_smiles_vs_market(vs, market)
        n_plots += 2

    print(f"surface assembly: {report['n_dates']} dates, {report['n_slices_total']} slices | "
          f"interp butterfly ok {report['all_interp_butterfly_ok']} "
          f"(worst g {report['worst_interp_min_g']:+.2e}) | "
          f"calendar ok {report['all_interp_calendar_ok']}")
    print(f"vs market: median RMSE {report['rmse_iv_median'] * 100:.2f} volpts | "
          f"max abs err {report['max_abs_err_iv'] * 100:.2f} | "
          f"within 1 volpt {report['frac_within_1volpt']:.1%}")
    print(f"-> {qc_path}")
    print(f"-> {n_plots} plots in {PLOTS_DIR}")
    return report


if __name__ == "__main__":
    run_assembly()
