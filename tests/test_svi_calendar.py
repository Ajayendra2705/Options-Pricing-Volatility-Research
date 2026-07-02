"""
Day 13 tests: no-calendar constraint (joint across slices).

- check_calendar detects a synthetic violating pair (long-T slice below
  short-T) with correct severity; clean pair passes.
- fit_svi_constrained with w_floor: refit clears the floor with bounded
  RMSE cost and stays butterfly-free.
- fit_all_joint on a synthetic violating surface produces a calendar-clean
  set of slices.
- Real surface: joint fit -> 0 butterfly violations, 0 calendar pairs
  violated; arb_violations.json structure complete.
"""

import numpy as np
import pandas as pd
import pytest

from src.surface.clean import PROCESSED_DIR
from src.surface.no_arb import (
    CAL_MARGIN,
    check_butterfly,
    check_calendar,
    fit_all_joint,
    fit_svi_constrained,
)
from src.surface.svi import SVIParams, svi_iv, svi_total_variance

SHORT = SVIParams(a=0.020, b=0.30, rho=-0.5, m=0.0, sigma=0.20)   # T=0.25
# mildly infeasible long slice: dips a touch below SHORT (realistic violation)
LONG_BAD = SVIParams(a=0.017, b=0.29, rho=-0.5, m=0.0, sigma=0.20)
LONG_GOOD = SVIParams(a=0.045, b=0.32, rho=-0.5, m=0.0, sigma=0.22)


def fits_frame(slices):
    rows = []
    for expiry, T, p in slices:
        rows.append({"date": pd.Timestamp("2023-06-02"), "expiry": expiry, "T": T,
                     "fit_ok": True, "a": p.a, "b": p.b, "rho": p.rho,
                     "m": p.m, "sigma": p.sigma})
    return pd.DataFrame(rows)


def test_check_calendar_detects_violation():
    cal = check_calendar(fits_frame([("2023-09-15", 0.25, SHORT),
                                     ("2023-12-15", 0.50, LONG_BAD)]))
    assert len(cal) == 1
    assert cal.loc[0, "violated"]
    assert cal.loc[0, "max_severity"] > 0
    # severity equals the true max decrease on the grid
    kg = np.linspace(-1.5, 1.5, 1001)
    true_sev = (svi_total_variance(kg, SHORT) - svi_total_variance(kg, LONG_BAD)).max()
    assert cal.loc[0, "max_severity"] == pytest.approx(true_sev, rel=1e-6)


def test_check_calendar_clean_pair():
    cal = check_calendar(fits_frame([("2023-09-15", 0.25, SHORT),
                                     ("2023-12-15", 0.50, LONG_GOOD)]))
    assert len(cal) == 1
    assert not cal.loc[0, "violated"]
    assert cal.loc[0, "max_severity"] == 0.0


def test_floor_constrained_refit_clears_floor():
    K = np.linspace(-0.6, 0.6, 25)
    T_long = 0.50
    iv_bad = svi_iv(K, T_long, LONG_BAD)               # market quotes violating calendar
    kg = np.linspace(-1.5, 1.5, 1001)
    floor = svi_total_variance(kg, SHORT)
    params, rep = fit_svi_constrained(K, iv_bad, T_long, n_constraint=1001, w_floor=floor)
    assert not rep["pre_floor_ok"]                     # violation confirmed pre-fit
    assert rep["constraint_active"]
    assert rep["floor_ok"], rep
    assert rep["arb_free"]
    w_fit = svi_total_variance(kg, params)
    assert (w_fit >= floor).all()
    # cost bounded: fitted curve within a few volpts of the (infeasible) quotes
    assert rep["rmse_iv"] < 0.05


def test_fit_all_joint_synthetic_violating_surface():
    rows = []
    for expiry, T, p in (("2023-09-15", 0.25, SHORT), ("2023-12-15", 0.50, LONG_BAD)):
        for k in np.linspace(-0.5, 0.5, 21):
            rows.append({"date": pd.Timestamp("2023-06-02"), "expiry": expiry, "T": T,
                         "strike": 100 * np.exp(k), "F": 100.0,
                         "option_type": "P" if k < 0 else "C",
                         "log_moneyness": k, "iv": float(svi_iv(k, T, p)),
                         "status": "ok"})
    fits = fit_all_joint(pd.DataFrame(rows))
    assert fits["fit_ok"].all()
    cal = check_calendar(fits)
    assert not cal["violated"].any(), cal.to_dict()
    assert fits["arb_free"].astype(bool).all()


# --- Real surface -------------------------------------------------------------

@pytest.fixture(scope="module")
def real_joint():
    path = PROCESSED_DIR / "iv_surface.parquet"
    if not path.exists():
        pytest.skip("iv surface not built")
    return fit_all_joint(pd.read_parquet(path))


def test_real_joint_butterfly_clean(real_joint):
    ok = real_joint[real_joint["fit_ok"] == True]  # noqa: E712
    assert len(ok) >= 12
    assert ok["arb_free"].astype(bool).all()
    assert ok["floor_ok"].astype(bool).all()


def test_real_joint_calendar_clean(real_joint):
    cal = check_calendar(real_joint)
    assert len(cal) >= 8                              # 5 dates, 2-3 expiries each
    assert not cal["violated"].any(), cal[cal["violated"]].to_dict()


def test_real_joint_rmse_cost_small(real_joint):
    ok = real_joint[real_joint["fit_ok"] == True]  # noqa: E712
    assert (ok["rmse_iv"].median() * 100) < 1.0
    # calendar floor shouldn't blow up any single slice
    assert (ok["rmse_iv"].max() * 100) < 3.0
