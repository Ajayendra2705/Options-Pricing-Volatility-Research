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
    run_butterfly_check()
