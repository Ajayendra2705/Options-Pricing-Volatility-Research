"""
Day 14 — Surface assembly + QC.

Stitch the arb-free joint SVI slices (Day 13) into a queryable vol surface
per quote date:

- In T, between expiry nodes: LINEAR interpolation of NORMALIZED OPTION
  PRICES at fixed forward moneyness k = ln(K/F_T), then invert back to
  total variance. A convex combination of two arb-free call-price curves
  is convex and calendar-ordered, so interpolated slices are statically
  arb-free BY CONSTRUCTION (linear-in-w has no such guarantee and produced
  a real interpolated butterfly violation on the 2023-06-09 surface).
- Beyond the last node: flat total variance, w(k,T) = w(k,T_last) — same
  slice, so butterfly-free and weakly calendar-monotone by construction
  (flat-IV scaling w*T/T_last broke Durrleman g at 1.25*T_last on real
  data). Conservative: no variance growth past the quoted range.
- Before the first node: flat-IV scaling w(k,T) = w(k,T_first)*T/T_first
  (scaling DOWN; no construction guarantee, so QC checks it numerically).
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

K_QC_SPAN = 1.0        # fallback QC range when a fit carries no quoted k range
N_T_QC = 21            # interpolated T slices per date in the QC scan

# Day 32: the QC domain is the k range where EVERY node slice of the date is
# quoted (their intersection), not a fixed span. Interpolating in T at a k
# where only one expiry has quotes means interpolating against an extrapolated
# wing — and the no-arb fit does not constrain the slices there (see the
# CAL_DOMAIN note in no_arb.py), so checking there measures the extrapolation,
# not the surface. v1 checked +-1.0 with data out to only ~+-0.25: it was
# claiming arb-freedom over 4x the range it had quotes for.


def _black_norm(k, w, cp):
    """Undiscounted normalized Black price (F=1, K=e^k, total variance w)."""
    from scipy.stats import norm
    s = np.sqrt(w)
    d1 = -k / s + s / 2.0
    return cp * (norm.cdf(cp * d1) - np.exp(k) * norm.cdf(cp * (d1 - s)))


def _w_price_interp(k, w1, w2, lam):
    """Total variance from linear-in-T interpolation of normalized OTM-side
    option prices at fixed k. Statically arb-free by construction: a convex
    combination of two convex, calendar-ordered price curves stays convex
    and ordered.

    Far wings where the time value underflows double precision invert
    degenerately; there ANY w in [w1, w2] reprices identically to machine
    precision, so linear-in-w is substituted (price-identical, arb-free at
    machine precision)."""
    from scipy.stats import norm

    k = np.asarray(k, float)
    cp = np.where(k < 0.0, -1, 1)                  # OTM side: absolute price,
    p1 = _black_norm(k, w1, cp)                    # not a difference of larges
    p2 = _black_norm(k, w2, cp)
    p = (1.0 - lam) * p1 + lam * p2
    lo, hi = np.minimum(w1, w2), np.maximum(w1, w2)
    # vectorized Newton in w: C is increasing and concave-free in w with
    # dC/dw = phi(d1)/(2 sqrt(w)); bracketed by [lo, hi], seeded at linear-w
    w = np.clip((1.0 - lam) * w1 + lam * w2, lo, hi)
    for _ in range(60):
        s = np.sqrt(w)
        d1 = -k / s + s / 2.0
        diff = _black_norm(k, w, cp) - p
        vega_w = norm.pdf(d1) / (2.0 * s)
        step = np.where(vega_w > 1e-30, diff / np.maximum(vega_w, 1e-300), 0.0)
        w_new = np.clip(w - step, lo, hi)
        if np.max(np.abs(w_new - w)) < 1e-15:
            w = w_new
            break
        w = w_new
    return w


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
    # k range quoted on EVERY node slice: the domain the surface is claimed on
    k_lo: float | None = None
    k_hi: float | None = None

    def w(self, k, T: float):
        """Total variance at forward moneyness k, time T (interp/extrap in T)."""
        k = np.asarray(k, float)
        Ts = self.Ts
        if T <= Ts[0]:
            return svi_total_variance(k, self.params[0]) * (T / Ts[0])
        if T >= Ts[-1]:
            return svi_total_variance(k, self.params[-1]) + 0.0 * k
        i = int(np.searchsorted(Ts, T, side="right") - 1)
        lam = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
        w1 = svi_total_variance(k, self.params[i])
        w2 = svi_total_variance(k, self.params[i + 1])
        return _w_price_interp(k, w1, w2, lam)

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
    has_range = {"k_lo", "k_hi"} <= set(ok.columns)
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
            # intersection: quoted on every node, so T-interpolation between any
            # pair of them is backed by quotes on both sides
            k_lo=float(g["k_lo"].max()) if has_range else None,
            k_hi=float(g["k_hi"].min()) if has_range else None,
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
    """QC one date: market residuals + arb checks on interpolated slices.

    The arb scan runs on the date's quoted k domain (see the K_QC_SPAN note);
    `k_span` is the fallback when the fits carry no quoted range.
    """
    if vs.k_lo is not None and vs.k_hi is not None:
        k_lo, k_hi = vs.k_lo, vs.k_hi
    else:
        k_lo, k_hi = -k_span, k_span
    kg = np.linspace(k_lo, k_hi, 801)

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
        "k_checked": [float(k_lo), float(k_hi)],
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


def plot_surface_3d(vs: VolSurface, out_dir: Path = PLOTS_DIR,
                    ticker: str = "AAPL") -> Path:
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
    ax.set_title(f"{ticker} SVI surface — {d}")
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


def run_assembly(
    fits_path: Path | None = None,
    processed_dir: Path | None = None,
    plots_dir: Path | None = None,
    qc_path: Path | None = None,
    make_plots: bool = True,
    ticker: str = "AAPL",
) -> dict:
    """Day 14 deliverable: surfaces + QC json + 3D/smile plots."""
    processed_dir = processed_dir or PROCESSED_DIR
    plots_dir = plots_dir or PLOTS_DIR
    fits = pd.read_parquet(fits_path or processed_dir / "svi_params_joint.parquet")
    forwards = pd.read_parquet(processed_dir / "forwards.parquet")
    market = pd.read_parquet(processed_dir / "iv_surface.parquet")

    surfaces = build_surfaces(fits, forwards)
    per_date = [qc_surface(vs, market) for vs in surfaces.values()]

    report = {
        "n_dates": len(per_date),
        "n_slices_total": int(sum(d["n_expiries"] for d in per_date)),
        # the arb claims below hold on this domain, per date: the log-moneyness
        # range quoted on every one of that date's expiries (Day 32)
        "arb_checked_on": "intersection of the date's quoted log-moneyness",
        "narrowest_k_checked": [
            float(max(d["k_checked"][0] for d in per_date)),
            float(min(d["k_checked"][1] for d in per_date)),
        ],
        "all_interp_butterfly_ok": bool(all(d["interp_butterfly_ok"] for d in per_date)),
        "all_interp_calendar_ok": bool(all(d["interp_calendar_ok"] for d in per_date)),
        "worst_interp_min_g": float(min(d["interp_min_g"] for d in per_date)),
        "rmse_iv_median": float(np.median([d["rmse_iv"] for d in per_date])),
        "max_abs_err_iv": float(max(d["max_abs_err_iv"] for d in per_date)),
        "frac_within_1volpt": float(np.mean([d["frac_within_1volpt"] for d in per_date])),
        "dates": per_date,
    }
    qc_path = qc_path or PROJECT_ROOT / "results" / "surface_qc.json"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(report, indent=2), newline="\n")

    # 155 SPY dates would emit 310 figures for a stage whose deliverable is the
    # QC json — plots stay on for v1 (14 tracked pngs), off for bulk runs.
    n_plots = 0
    if make_plots:
        for vs in surfaces.values():
            plot_surface_3d(vs, plots_dir, ticker)
            plot_smiles_vs_market(vs, market, plots_dir)
            n_plots += 2

    print(f"surface assembly: {report['n_dates']} dates, {report['n_slices_total']} slices | "
          f"interp butterfly ok {report['all_interp_butterfly_ok']} "
          f"(worst g {report['worst_interp_min_g']:+.2e}) | "
          f"calendar ok {report['all_interp_calendar_ok']}")
    print(f"vs market: median RMSE {report['rmse_iv_median'] * 100:.2f} volpts | "
          f"max abs err {report['max_abs_err_iv'] * 100:.2f} | "
          f"within 1 volpt {report['frac_within_1volpt']:.1%}")
    print(f"-> {qc_path}")
    print(f"-> {n_plots} plots in {plots_dir}")
    return report


if __name__ == "__main__":
    run_assembly()
