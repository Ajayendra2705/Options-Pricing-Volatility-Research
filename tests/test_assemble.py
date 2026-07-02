"""
Day 14 tests: surface assembly + QC.

Synthetic: two arb-free calendar-ordered slices ->
- exact recovery at node expiries;
- linear-in-w interpolation between nodes;
- flat-IV extrapolation outside;
- calendar monotone in T everywhere;
- forward curve: ln F linear in T, iv_strike consistent with iv(k).
Real: assembled surfaces pass interpolated butterfly + calendar QC,
market residuals small, QC json structure complete.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.surface.assemble import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    VolSurface,
    _g_fd,
    build_surfaces,
    qc_surface,
)
from src.surface.svi import SVIParams, svi_iv, svi_total_variance

P_SHORT = SVIParams(a=0.020, b=0.30, rho=-0.5, m=0.0, sigma=0.20)   # T=0.25
P_LONG = SVIParams(a=0.045, b=0.32, rho=-0.5, m=0.0, sigma=0.22)    # T=0.50
K = np.linspace(-0.8, 0.8, 41)


@pytest.fixture
def vs():
    return VolSurface(date=pd.Timestamp("2023-06-02"),
                      Ts=np.array([0.25, 0.50]), params=[P_SHORT, P_LONG],
                      F_nodes=np.array([100.0, 101.0]),
                      expiries=["2023-09-15", "2023-12-15"])


def test_nodes_recovered_exactly(vs):
    np.testing.assert_allclose(vs.iv(K, 0.25), svi_iv(K, 0.25, P_SHORT), rtol=1e-14)
    np.testing.assert_allclose(vs.iv(K, 0.50), svi_iv(K, 0.50, P_LONG), rtol=1e-14)


def test_linear_w_interpolation_between_nodes(vs):
    w_mid = vs.w(K, 0.375)
    expected = 0.5 * (svi_total_variance(K, P_SHORT) + svi_total_variance(K, P_LONG))
    np.testing.assert_allclose(w_mid, expected, rtol=1e-14)


def test_flat_iv_extrapolation(vs):
    # short end: same IV as first node; long end: same IV as last node
    np.testing.assert_allclose(vs.iv(K, 0.10), svi_iv(K, 0.25, P_SHORT), rtol=1e-14)
    np.testing.assert_allclose(vs.iv(K, 0.80), svi_iv(K, 0.50, P_LONG), rtol=1e-14)


def test_calendar_monotone_everywhere(vs):
    t_grid = np.linspace(0.05, 1.0, 40)
    w_stack = np.array([vs.w(K, t) for t in t_grid])
    assert (np.diff(w_stack, axis=0) >= -1e-14).all()


def test_interpolated_slices_butterfly_free(vs):
    kg = np.linspace(-1.0, 1.0, 801)
    for t in np.linspace(0.10, 0.80, 15):
        assert _g_fd(lambda kk: vs.w(kk, t), kg).min() >= 0


def test_forward_lnF_linear(vs):
    # geometric midpoint between nodes
    assert vs.forward(0.375) == pytest.approx(np.sqrt(100.0 * 101.0), rel=1e-12)
    assert vs.forward(0.25) == pytest.approx(100.0, rel=1e-12)
    # extrapolation continues the edge carry: lnF slope = ln(101/100)/0.25
    assert vs.forward(0.75) == pytest.approx(101.0 * (101.0 / 100.0), rel=1e-12)


def test_iv_strike_consistent(vs):
    T = 0.375
    F = vs.forward(T)
    Ks = F * np.exp(K)
    np.testing.assert_allclose(vs.iv_strike(Ks, T), vs.iv(K, T), rtol=1e-14)


def test_build_surfaces_wires_forwards():
    fits = pd.DataFrame([
        {"date": pd.Timestamp("2023-06-02"), "expiry": "2023-09-15", "T": 0.25,
         "fit_ok": True, "a": P_SHORT.a, "b": P_SHORT.b, "rho": P_SHORT.rho,
         "m": P_SHORT.m, "sigma": P_SHORT.sigma},
        {"date": pd.Timestamp("2023-06-02"), "expiry": "2023-12-15", "T": 0.50,
         "fit_ok": True, "a": P_LONG.a, "b": P_LONG.b, "rho": P_LONG.rho,
         "m": P_LONG.m, "sigma": P_LONG.sigma},
    ])
    forwards = pd.DataFrame([
        {"date": pd.Timestamp("2023-06-02"), "expiry": "2023-09-15", "F": 100.0},
        {"date": pd.Timestamp("2023-06-02"), "expiry": "2023-12-15", "F": 101.0},
    ])
    surfs = build_surfaces(fits, forwards)
    assert len(surfs) == 1
    s = surfs[pd.Timestamp("2023-06-02")]
    np.testing.assert_allclose(s.F_nodes, [100.0, 101.0])
    np.testing.assert_allclose(s.Ts, [0.25, 0.50])


# --- Real surface -------------------------------------------------------------

@pytest.fixture(scope="module")
def real_surfaces():
    fits_path = PROCESSED_DIR / "svi_params_joint.parquet"
    if not fits_path.exists():
        pytest.skip("joint fits not built")
    fits = pd.read_parquet(fits_path)
    forwards = pd.read_parquet(PROCESSED_DIR / "forwards.parquet")
    market = pd.read_parquet(PROCESSED_DIR / "iv_surface.parquet")
    return build_surfaces(fits, forwards), market


def test_real_interp_arb_free(real_surfaces):
    surfaces, market = real_surfaces
    assert len(surfaces) == 5
    for vs in surfaces.values():
        qc = qc_surface(vs, market)
        assert qc["interp_butterfly_ok"], qc
        assert qc["interp_calendar_ok"], qc


def test_real_market_residuals_small(real_surfaces):
    surfaces, market = real_surfaces
    rmses = [qc_surface(vs, market)["rmse_iv"] for vs in surfaces.values()]
    assert np.median(rmses) * 100 < 1.0
    assert max(rmses) * 100 < 2.0


def test_real_qc_json_complete():
    path = PROJECT_ROOT / "results" / "surface_qc.json"
    if not path.exists():
        pytest.skip("qc json not built")
    rep = json.loads(path.read_text())
    for key in ("n_dates", "n_slices_total", "all_interp_butterfly_ok",
                "all_interp_calendar_ok", "rmse_iv_median", "dates"):
        assert key in rep
    assert rep["n_dates"] == len(rep["dates"]) == 5
    assert rep["all_interp_butterfly_ok"] and rep["all_interp_calendar_ok"]
    for d in rep["dates"]:
        for key in ("date", "n_expiries", "rmse_iv", "interp_min_g",
                    "frac_within_1volpt"):
            assert key in d
