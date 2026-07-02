"""
Day 11 — No-butterfly (Durrleman) condition for SVI slices.

A total-variance slice w(k) is free of butterfly arbitrage iff Durrleman's

    g(k) = (1 - k*w'/(2w))^2 - (w'^2/4) * (1/w + 1/4) + w''/2  >=  0

for all k (and w > 0). g(k) is proportional to the risk-neutral density,
so g < 0 anywhere means a negative density — a butterfly you could buy for
credit. For raw SVI the derivatives are analytic:

    d = k - m,  R = sqrt(d^2 + sigma^2)
    w'(k)  = b * (rho + d / R)
    w''(k) = b * sigma^2 / R^3

Day 11 scope: DETECTION only (violations found, logged, quantified).
Refit under the constraint is Day 12; calendar (inter-slice) is Day 13.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.surface.svi import SVIParams, svi_total_variance

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

K_SPAN = 1.5          # default check range in log-moneyness
N_GRID = 1001


def _unpack(params) -> tuple[float, float, float, float, float]:
    if isinstance(params, SVIParams):
        return params.a, params.b, params.rho, params.m, params.sigma
    a, b, rho, m, sigma = np.asarray(params, float)
    return a, b, rho, m, sigma


def svi_w_prime(k, params):
    """Analytic w'(k) for raw SVI."""
    a, b, rho, m, sigma = _unpack(params)
    d = np.asarray(k, float) - m
    return b * (rho + d / np.sqrt(d**2 + sigma**2))


def svi_w_second(k, params):
    """Analytic w''(k) for raw SVI."""
    a, b, rho, m, sigma = _unpack(params)
    d = np.asarray(k, float) - m
    return b * sigma**2 / (d**2 + sigma**2) ** 1.5


def durrleman_g(k, params):
    """Durrleman g(k). Requires w(k) > 0; returns -inf where w <= 0."""
    k = np.asarray(k, float)
    w = svi_total_variance(k, params)
    wp = svi_w_prime(k, params)
    wpp = svi_w_second(k, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * wp / (2.0 * w)) ** 2 - (wp**2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0
    return np.where(w > 0, g, -np.inf)


def check_butterfly(params, k_range: tuple[float, float] = (-K_SPAN, K_SPAN),
                    n: int = N_GRID) -> dict:
    """Scan g(k) on a grid. Returns violation report for one slice."""
    k = np.linspace(k_range[0], k_range[1], n)
    g = durrleman_g(k, params)
    w = svi_total_variance(k, params)
    viol = g < 0
    return {
        "arb_free": bool(not viol.any()),
        "min_g": float(g.min()),
        "k_at_min_g": float(k[np.argmin(g)]),
        "n_violations": int(viol.sum()),
        "frac_violating": float(viol.mean()),
        "min_w": float(w.min()),
        "k_range": [float(k_range[0]), float(k_range[1])],
    }


def check_all_slices(fits: pd.DataFrame, k_span: float = K_SPAN) -> pd.DataFrame:
    """Butterfly check for every fitted slice of a Day-10 params table."""
    rows = []
    for _, f in fits[fits["fit_ok"]].iterrows():
        rep = check_butterfly((f["a"], f["b"], f["rho"], f["m"], f["sigma"]),
                              (-k_span, k_span))
        rows.append({"date": f["date"], "expiry": f["expiry"], "T": f["T"], **rep})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Day 12 — constrained refit: butterfly-free SVI via SLSQP (+ penalty fallback)
# ---------------------------------------------------------------------------

W_FLOOR = 1e-8
G_MARGIN = 2e-4        # require g >= margin: SLSQP satisfies constraints only
                       # to ~ftol, so leave room for the stricter post-check


def _g_finite(k, p):
    """Durrleman g with w clipped at W_FLOOR so SLSQP sees finite values.
    Where w is genuinely >= floor this equals durrleman_g exactly."""
    k = np.asarray(k, float)
    w = np.maximum(svi_total_variance(k, p), W_FLOOR)
    from src.surface.no_arb import svi_w_prime, svi_w_second  # self, explicit
    wp = svi_w_prime(k, p)
    wpp = svi_w_second(k, p)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp**2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


def fit_svi_constrained(k, iv, T, k_span: float = K_SPAN, n_constraint: int = 201):
    """Fit raw SVI under the no-butterfly constraint.

    1. Unconstrained fit (Day 9 machinery). If already arb-free on the
       constraint grid -> return it (constraint_active=False).
    2. Else SLSQP: minimize sum of squared total-variance residuals s.t.
       g(k_i) >= G_MARGIN and w(k_i) >= W_FLOOR on the grid.
    3. If SLSQP fails/violates: escalating soft-hinge penalty fallback.

    Returns (params, report). report['arb_free'] is the post-fit verdict.
    """
    from scipy.optimize import minimize

    from src.surface.svi import SVIParams, fit_svi_slice

    k = np.asarray(k, float)
    w_mkt = np.asarray(iv, float) ** 2 * T
    kg = np.linspace(-k_span, k_span, n_constraint)

    params0, rep0 = fit_svi_slice(k, iv, T)
    pre = check_butterfly(params0, (-k_span, k_span))
    base = {
        "rmse_iv_unconstrained": rep0["rmse_iv"],
        "pre_arb_free": pre["arb_free"],
        "pre_min_g": pre["min_g"],
    }
    if pre["arb_free"]:
        post = check_butterfly(params0, (-k_span, k_span))
        return params0, {**base, "constraint_active": False, "method": "unconstrained",
                         "arb_free": True, "min_g": post["min_g"],
                         "rmse_iv": rep0["rmse_iv"]}

    def obj(p):
        return float(np.sum((svi_total_variance(k, p) - w_mkt) ** 2))

    span = max(k.max() - k.min(), 1e-3)
    w_max = w_mkt.max()
    bounds = [(-w_max, 2 * w_max + 1e-8), (1e-8, 10.0 * (w_max / span + 1.0)),
              (-0.999, 0.999), (k.min() - span, k.max() + span), (1e-4, 4.0 * span + 1.0)]
    cons = [{"type": "ineq", "fun": lambda p: _g_finite(kg, p) - G_MARGIN},
            {"type": "ineq", "fun": lambda p: svi_total_variance(kg, p) - W_FLOOR}]

    sol = minimize(obj, params0.as_array(), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    cand, method = sol.x, "slsqp"

    post = check_butterfly(cand, (-k_span, k_span))
    if not post["arb_free"]:
        # penalty fallback: escalate hinge weight until clean
        for lam in (1e2, 1e4, 1e6):
            def pobj(p, lam=lam):
                gviol = np.minimum(_g_finite(kg, p) - G_MARGIN, 0.0)
                wviol = np.minimum(svi_total_variance(kg, p) - W_FLOOR, 0.0)
                return obj(p) + lam * (np.sum(gviol**2) + np.sum(wviol**2))
            sol = minimize(pobj, cand, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 1000})
            cand = sol.x
            post = check_butterfly(cand, (-k_span, k_span))
            if post["arb_free"]:
                method = f"penalty_{lam:g}"
                break

    params = SVIParams(*cand)
    w_fit = svi_total_variance(k, params)
    rmse_iv = float(np.sqrt(np.mean((np.sqrt(np.maximum(w_fit, 0.0) / T) - np.asarray(iv, float)) ** 2)))
    return params, {**base, "constraint_active": True, "method": method,
                    "arb_free": post["arb_free"], "min_g": post["min_g"],
                    "rmse_iv": rmse_iv}


def refit_all_constrained(surf: pd.DataFrame, k_span: float = K_SPAN) -> pd.DataFrame:
    """Constrained fit for every slice; one row per slice incl. violation log."""
    from src.surface.svi import MIN_POINTS, otm_side

    rows = []
    for (date, expiry), g in surf.groupby(["date", "expiry"]):
        sl = otm_side(g)
        if len(sl) < MIN_POINTS:
            rows.append({"date": date, "expiry": expiry, "n_points": len(sl), "fit_ok": False})
            continue
        T = float(sl["T"].iloc[0])
        params, rep = fit_svi_constrained(sl["log_moneyness"], sl["iv"], T, k_span)
        rows.append({"date": date, "expiry": expiry, "n_points": len(sl), "fit_ok": True,
                     "T": T, "a": params.a, "b": params.b, "rho": params.rho,
                     "m": params.m, "sigma": params.sigma, **rep})
    return pd.DataFrame(rows).sort_values(["date", "expiry"]).reset_index(drop=True)


def run_constrained_refit(surface_path: Path | None = None) -> pd.DataFrame:
    """Day 12 deliverable: arb-free per-slice fits + violation log."""
    import json

    surf = pd.read_parquet(surface_path or PROCESSED_DIR / "iv_surface.parquet")
    fits = refit_all_constrained(surf)
    ok = fits[fits["fit_ok"]]

    out = PROCESSED_DIR / "svi_params_constrained.parquet"
    fits.to_parquet(out, index=False)

    log = {
        "n_slices": int(len(fits)),
        "n_fitted": int(len(ok)),
        "n_pre_violations": int((~ok["pre_arb_free"].astype(bool)).sum()),
        "n_post_violations": int((~ok["arb_free"].astype(bool)).sum()),
        "slices": [
            {"date": str(pd.Timestamp(r["date"]).date()),
             "expiry": str(pd.Timestamp(r["expiry"]).date()),
             "pre_arb_free": bool(r["pre_arb_free"]), "arb_free": bool(r["arb_free"]),
             "method": r["method"], "min_g": float(r["min_g"]),
             "rmse_unconstrained": float(r["rmse_iv_unconstrained"]),
             "rmse_constrained": float(r["rmse_iv"])}
            for _, r in ok.iterrows()
        ],
    }
    log_path = PROJECT_ROOT / "results" / "svi_butterfly_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    print(f"constrained refit: {len(ok)}/{len(fits)} fitted | "
          f"pre-violations {log['n_pre_violations']} | post-violations {log['n_post_violations']}")
    print(f"median RMSE {ok['rmse_iv'].median() * 100:.2f} volpts "
          f"(unconstrained {ok['rmse_iv_unconstrained'].median() * 100:.2f})")
    print(f"-> {out}")
    print(f"-> {log_path}")
    return fits


def run_butterfly_check(params_path: Path | None = None) -> pd.DataFrame:
    fits = pd.read_parquet(params_path or PROCESSED_DIR / "svi_params.parquet")
    rep = check_all_slices(fits)
    n_bad = int((~rep["arb_free"]).sum())
    print(f"butterfly check: {len(rep)} slices | violations on {n_bad}")
    tab = rep.assign(date=lambda d: pd.to_datetime(d["date"]).dt.date,
                     expiry=lambda d: pd.to_datetime(d["expiry"]).dt.date)
    print(tab[["date", "expiry", "T", "arb_free", "min_g", "k_at_min_g"]].to_string(index=False))
    return rep


if __name__ == "__main__":
    import sys

    if "--refit" in sys.argv:
        run_constrained_refit()
    else:
        run_butterfly_check()
