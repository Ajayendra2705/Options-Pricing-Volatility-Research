"""
v2 robustness — SABR second calibration (independent surface family).

The whole signal rests on ATM implied vol read off the joint arb-free SVI fit.
A fair robustness question is whether that number is an artifact of the SVI
functional form. SABR (Hagan 2002) is a different, economically-motivated
parametric family; if it fits the SAME quoted points about as well and lands on
the SAME ATM mark, the surface — and the signal built on it — is not
SVI-specific.

We use the lognormal case beta = 1 (standard for index options): it fixes the
CEV backbone and leaves (alpha, rho, nu) free — three shape parameters, matching
the effective freedom of SVI's smile — so the comparison is like for like. The
fit minimises RMSE in implied-vol space on the same OTM points SVI uses
(svi.fit_points), and is deliberately UNCONSTRAINED: we are testing agreement of
two independent fits, not re-imposing no-arbitrage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.surface.svi import MIN_POINTS, fit_points

BETA_DEFAULT = 1.0


def sabr_iv(F, K, T, alpha: float, rho: float, nu: float,
            beta: float = BETA_DEFAULT):
    """Hagan (2002) lognormal-implied vol for SABR. Vectorised over K.

    F: forward (scalar), K: strike(s), T: year fraction. Returns Black IV.
    ATM (K == F) uses the limiting expansion; |z| -> 0 gives z/x(z) -> 1.
    """
    F = float(F)
    K = np.asarray(K, float)
    one_mb = 1.0 - beta
    fk_beta = (F * K) ** (one_mb / 2.0)
    log_fk = np.log(F / K)

    # z / x(z), with the ATM limit handled continuously
    z = (nu / alpha) * fk_beta * log_fk
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    x = np.log((sqrt_term + z - rho) / (1.0 - rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_over_x = np.where(np.abs(z) < 1e-8, 1.0, z / x)

    denom = fk_beta * (1.0
                       + (one_mb ** 2 / 24.0) * log_fk ** 2
                       + (one_mb ** 4 / 1920.0) * log_fk ** 4)
    correction = 1.0 + (
        (one_mb ** 2 / 24.0) * alpha ** 2 / (F * K) ** one_mb
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2) * T
    return (alpha / denom) * z_over_x * correction


def fit_sabr_slice(F: float, K, iv, T: float,
                   beta: float = BETA_DEFAULT) -> tuple[dict, dict]:
    """Least-squares SABR fit of one slice in implied-vol space.

    Returns (params, report) with params = {alpha, beta, rho, nu} and report
    holding rmse_iv, n_points, atm_iv (the beta-fixed ATM mark, K = F).
    """
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)

    def resid(p):
        alpha, rho, nu = p
        return sabr_iv(F, K, T, alpha, rho, nu, beta) - iv

    span_iv = max(float(np.mean(iv)), 1e-3)
    lb = [1e-4, -0.999, 1e-4]
    ub = [4.0, 0.999, 20.0]
    best, best_cost = None, np.inf
    for a0 in (span_iv * 0.5, span_iv, span_iv * 1.5):
        for r0 in (-0.5, 0.0, 0.5):
            for n0 in (0.3, 1.0, 2.0):
                p0 = np.clip([a0, r0, n0], lb, ub)
                try:
                    sol = least_squares(resid, p0, bounds=(lb, ub),
                                        method="trf", max_nfev=2000)
                except ValueError:
                    continue
                if sol.cost < best_cost:
                    best, best_cost = sol, sol.cost
    if best is None:
        raise RuntimeError("SABR fit failed from every start")

    alpha, rho, nu = best.x
    iv_fit = sabr_iv(F, K, T, alpha, rho, nu, beta)
    params = {"alpha": float(alpha), "beta": float(beta),
              "rho": float(rho), "nu": float(nu)}
    report = {
        "rmse_iv": float(np.sqrt(np.mean((iv_fit - iv) ** 2))),
        "n_points": int(K.size),
        "atm_iv": float(sabr_iv(F, F, T, alpha, rho, nu, beta)),
    }
    return params, report


def fit_all_slices_sabr(surf: pd.DataFrame,
                        beta: float = BETA_DEFAULT) -> pd.DataFrame:
    """SABR fit for every (date, expiry) slice, on the same OTM points as SVI."""
    rows = []
    for (date, expiry), g in surf.groupby(["date", "expiry"]):
        sl, augmented = fit_points(g)
        row = {"date": date, "expiry": expiry, "n_points": len(sl),
               "augmented": augmented}
        if len(sl) < MIN_POINTS:
            rows.append({**row, "fit_ok": False})
            continue
        T = float(sl["T"].iloc[0])
        F = float(sl["F"].iloc[0])
        try:
            params, report = fit_sabr_slice(F, sl["strike"], sl["iv"], T, beta)
        except RuntimeError:
            rows.append({**row, "fit_ok": False, "T": T})
            continue
        rows.append({**row, "fit_ok": True, "T": T, **params,
                     "rmse_iv": report["rmse_iv"], "atm_iv": report["atm_iv"]})
    out = pd.DataFrame(rows).sort_values(["date", "expiry"]).reset_index(drop=True)
    out["fit_ok"] = out["fit_ok"].astype(bool)
    return out
