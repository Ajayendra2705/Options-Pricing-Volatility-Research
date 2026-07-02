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


CAL_MARGIN = 1e-5      # min total-variance gap between consecutive expiries


def _floor_ok(p, kg, w_floor) -> bool:
    if w_floor is None:
        return True
    return bool((svi_total_variance(kg, p) >= w_floor).all())


def fit_svi_constrained(k, iv, T, k_span: float = K_SPAN, n_constraint: int = 201,
                        w_floor=None):
    """Fit raw SVI under no-butterfly (and optional calendar-floor) constraints.

    w_floor: optional array on the internal constraint grid
    linspace(-k_span, k_span, n_constraint) — the previous expiry's total
    variance; enforces w(k) >= w_floor + CAL_MARGIN (calendar no-arb).

    1. Unconstrained fit (Day 9 machinery). If already feasible -> done.
    2. Else SLSQP: min sum sq total-variance residuals s.t. g >= G_MARGIN,
       w >= W_FLOOR, and the calendar floor, all on the grid.
    3. If SLSQP fails/violates: escalating soft-hinge penalty fallback.

    Returns (params, report). report['arb_free'] is the post-fit butterfly
    verdict; report['floor_ok'] the calendar-floor verdict.
    """
    from scipy.optimize import minimize

    from src.surface.svi import SVIParams, fit_svi_slice

    k = np.asarray(k, float)
    w_mkt = np.asarray(iv, float) ** 2 * T
    kg = np.linspace(-k_span, k_span, n_constraint)
    if w_floor is not None:
        w_floor = np.asarray(w_floor, float)
        assert w_floor.shape == kg.shape, "w_floor must live on the constraint grid"

    params0, rep0 = fit_svi_slice(k, iv, T)
    pre = check_butterfly(params0, (-k_span, k_span))
    base = {
        "rmse_iv_unconstrained": rep0["rmse_iv"],
        "pre_arb_free": pre["arb_free"],
        "pre_min_g": pre["min_g"],
        "pre_floor_ok": _floor_ok(params0.as_array(), kg, w_floor),
    }
    if pre["arb_free"] and base["pre_floor_ok"]:
        return params0, {**base, "constraint_active": False, "method": "unconstrained",
                         "arb_free": True, "floor_ok": True, "min_g": pre["min_g"],
                         "rmse_iv": rep0["rmse_iv"]}

    def obj(p):
        return float(np.sum((svi_total_variance(k, p) - w_mkt) ** 2))

    span = max(k.max() - k.min(), 1e-3)
    w_max = max(w_mkt.max(), float(w_floor.max()) if w_floor is not None else 0.0)
    bounds = [(-w_max, 2 * w_max + 1e-8), (1e-8, 10.0 * (w_max / span + 1.0)),
              (-0.999, 0.999), (k.min() - span, k.max() + span), (1e-4, 4.0 * span + 1.0)]
    cons = [{"type": "ineq", "fun": lambda p: _g_finite(kg, p) - G_MARGIN},
            {"type": "ineq", "fun": lambda p: svi_total_variance(kg, p) - W_FLOOR}]
    if w_floor is not None:
        cons.append({"type": "ineq",
                     "fun": lambda p: svi_total_variance(kg, p) - w_floor - CAL_MARGIN})

    sol = minimize(obj, params0.as_array(), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    cand, method = sol.x, "slsqp"

    def feasible(p):
        return check_butterfly(p, (-k_span, k_span))["arb_free"] and _floor_ok(p, kg, w_floor)

    if not feasible(cand):
        # penalty fallback: escalate hinge weight until clean
        for lam in (1e2, 1e4, 1e6):
            def pobj(p, lam=lam):
                gviol = np.minimum(_g_finite(kg, p) - G_MARGIN, 0.0)
                wviol = np.minimum(svi_total_variance(kg, p) - W_FLOOR, 0.0)
                pen = np.sum(gviol**2) + np.sum(wviol**2)
                if w_floor is not None:
                    cviol = np.minimum(svi_total_variance(kg, p) - w_floor - CAL_MARGIN, 0.0)
                    pen += np.sum(cviol**2)
                return obj(p) + lam * pen
            sol = minimize(pobj, cand, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 1000})
            cand = sol.x
            if feasible(cand):
                method = f"penalty_{lam:g}"
                break

    params = SVIParams(*cand)
    post = check_butterfly(cand, (-k_span, k_span))
    w_fit = svi_total_variance(k, params)
    rmse_iv = float(np.sqrt(np.mean((np.sqrt(np.maximum(w_fit, 0.0) / T) - np.asarray(iv, float)) ** 2)))
    return params, {**base, "constraint_active": True, "method": method,
                    "arb_free": post["arb_free"], "floor_ok": _floor_ok(cand, kg, w_floor),
                    "min_g": post["min_g"], "rmse_iv": rmse_iv}


def refit_all_constrained(surf: pd.DataFrame, k_span: float = K_SPAN) -> pd.DataFrame:
    """Constrained fit for every slice; one row per slice incl. violation log."""
    from src.surface.svi import MIN_POINTS, fit_points

    rows = []
    for (date, expiry), g in surf.groupby(["date", "expiry"]):
        sl, augmented = fit_points(g)
        if len(sl) < MIN_POINTS:
            rows.append({"date": date, "expiry": expiry, "n_points": len(sl),
                         "augmented": augmented, "fit_ok": False})
            continue
        T = float(sl["T"].iloc[0])
        params, rep = fit_svi_constrained(sl["log_moneyness"], sl["iv"], T, k_span)
        rows.append({"date": date, "expiry": expiry, "n_points": len(sl),
                     "augmented": augmented, "fit_ok": True,
                     "T": T, "a": params.a, "b": params.b, "rho": params.rho,
                     "m": params.m, "sigma": params.sigma, **rep})
    out = pd.DataFrame(rows).sort_values(["date", "expiry"]).reset_index(drop=True)
    out["fit_ok"] = out["fit_ok"].astype(bool)      # never object dtype
    return out


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


# ---------------------------------------------------------------------------
# Day 13 — no-calendar constraint (joint across slices per quote date)
# ---------------------------------------------------------------------------
# Coordinate note: slices are parameterized in per-expiry forward moneyness
# k = ln(K/F_T) (Gatheral convention). Under the martingale measure the
# calendar condition is monotonicity of total variance in T at fixed k in
# THIS coordinate: w(k, T2) >= w(k, T1) for T2 > T1.


def check_calendar(fits: pd.DataFrame, k_span: float = K_SPAN,
                   n: int = N_GRID) -> pd.DataFrame:
    """Pairwise calendar check for consecutive expiries per quote date.

    Returns one row per (date, T_short, T_long) pair with the max total-
    variance decrease (severity, in variance units) and violating fraction.
    """
    kg = np.linspace(-k_span, k_span, n)
    rows = []
    ok = fits[fits["fit_ok"] == True]  # noqa: E712 (object dtype)
    for date, g in ok.groupby("date"):
        g = g.sort_values("T")
        ws = [svi_total_variance(kg, (f.a, f.b, f.rho, f.m, f.sigma))
              for f in g.itertuples()]
        for i in range(len(ws) - 1):
            dec = ws[i] - ws[i + 1]                     # >0 where calendar violated
            rows.append({
                "date": date,
                "T_short": float(g["T"].iloc[i]), "T_long": float(g["T"].iloc[i + 1]),
                "expiry_short": g["expiry"].iloc[i], "expiry_long": g["expiry"].iloc[i + 1],
                "max_severity": float(max(dec.max(), 0.0)),
                "frac_violating": float((dec > 0).mean()),
                "k_at_max": float(kg[np.argmax(dec)]),
                "violated": bool(dec.max() > 0),
            })
    return pd.DataFrame(rows)


def fit_all_joint(surf: pd.DataFrame, k_span: float = K_SPAN,
                  n_constraint: int = N_GRID) -> pd.DataFrame:
    """Joint (butterfly + calendar) fit: per date, fit slices short->long T,
    each floored by the previous slice's total variance on the grid.

    n_constraint defaults to the CHECK grid density (N_GRID): a coarser
    fit grid lets sub-node violations slip through the stricter check
    (caught on real data: 201-pt fit grid vs 1001-pt check grid)."""
    from src.surface.svi import MIN_POINTS, fit_points

    kg = np.linspace(-k_span, k_span, n_constraint)
    rows = []
    for date, gd in surf.groupby("date"):
        floor = None
        # order expiries by T
        slices = sorted(gd.groupby("expiry"), key=lambda kv: kv[1]["T"].iloc[0])
        for expiry, g in slices:
            sl, augmented = fit_points(g)
            if len(sl) < MIN_POINTS:
                rows.append({"date": date, "expiry": expiry, "n_points": len(sl),
                             "augmented": augmented, "fit_ok": False})
                continue
            T = float(sl["T"].iloc[0])
            params, rep = fit_svi_constrained(sl["log_moneyness"], sl["iv"], T,
                                              k_span, n_constraint, w_floor=floor)
            rows.append({"date": date, "expiry": expiry, "n_points": len(sl),
                         "augmented": augmented, "fit_ok": True, "T": T,
                         "a": params.a, "b": params.b,
                         "rho": params.rho, "m": params.m, "sigma": params.sigma, **rep})
            # next slice must clear this one (only if this fit is usable)
            if rep["arb_free"] and rep["floor_ok"]:
                floor = svi_total_variance(kg, params)
    out = pd.DataFrame(rows).sort_values(["date", "expiry"]).reset_index(drop=True)
    out["fit_ok"] = out["fit_ok"].astype(bool)      # never object dtype
    return out


def run_arb_check(surface_path: Path | None = None) -> dict:
    """Day 13 deliverable: joint fit + full violation report ->
    results/arb_violations.json (counts, max severity)."""
    import json

    surf = pd.read_parquet(surface_path or PROCESSED_DIR / "iv_surface.parquet")
    fits = fit_all_joint(surf)
    ok = fits[fits["fit_ok"] == True]  # noqa: E712

    out = PROCESSED_DIR / "svi_params_joint.parquet"
    fits.to_parquet(out, index=False)

    cal = check_calendar(fits)
    bfly_bad = int((~ok["arb_free"].astype(bool)).sum())
    floor_bad = int((~ok["floor_ok"].astype(bool)).sum())
    report = {
        "n_slices_fitted": int(len(ok)),
        "butterfly": {
            "n_violations": bfly_bad,
            "min_g_across_slices": float(ok["min_g"].min()) if len(ok) else None,
        },
        "calendar": {
            "n_pairs_checked": int(len(cal)),
            "n_pairs_violated": int(cal["violated"].sum()) if len(cal) else 0,
            "max_severity_w": float(cal["max_severity"].max()) if len(cal) else 0.0,
            "n_floor_failures_in_fit": floor_bad,
        },
        "rmse_iv_median": float(ok["rmse_iv"].median()) if len(ok) else None,
        "pairs": [
            {"date": str(pd.Timestamp(r["date"]).date()),
             "expiry_short": str(pd.Timestamp(r["expiry_short"]).date()),
             "expiry_long": str(pd.Timestamp(r["expiry_long"]).date()),
             "max_severity": r["max_severity"], "frac_violating": r["frac_violating"],
             "violated": r["violated"]}
            for _, r in cal.iterrows()
        ],
    }
    path = PROJECT_ROOT / "results" / "arb_violations.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"joint fit: {len(ok)} slices | butterfly violations {bfly_bad} | "
          f"calendar pairs violated {report['calendar']['n_pairs_violated']}"
          f"/{report['calendar']['n_pairs_checked']} "
          f"(max severity {report['calendar']['max_severity_w']:.2e}) | "
          f"median RMSE {report['rmse_iv_median'] * 100:.2f} volpts")
    print(f"-> {out}")
    print(f"-> {path}")
    return report


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

    if "--joint" in sys.argv:
        run_arb_check()
    elif "--refit" in sys.argv:
        run_constrained_refit()
    else:
        run_butterfly_check()
