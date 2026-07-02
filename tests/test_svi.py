"""
Day 9 tests: raw SVI single-slice calibration.

- Exact synthetic recovery: w generated from known SVI params -> fit
  recovers the curve (curve-identical even if params trade off).
- Noise robustness: RMSE tracks the injected noise scale.
- OTM-side selection helper.
- Real slice: fit converges, RMSE below threshold, w >= 0 on quoted range.
"""

import numpy as np
import pandas as pd
import pytest

from src.surface.clean import PROCESSED_DIR
from src.surface.iv_surface import build_iv_surface  # noqa: F401  (path check)
from src.surface.svi import (
    SVIParams,
    fit_all_slices,
    fit_svi_slice,
    otm_side,
    svi_iv,
    svi_total_variance,
)

TRUE = SVIParams(a=0.015, b=0.35, rho=-0.65, m=0.05, sigma=0.25)
T = 0.35
K_GRID = np.linspace(-0.35, 0.35, 25)


def test_svi_formula_known_value():
    # hand-computed: k=m -> w = a + b*sigma
    assert svi_total_variance(TRUE.m, TRUE) == pytest.approx(TRUE.a + TRUE.b * TRUE.sigma)
    # asymptotic wing slopes: right -> b*(1+rho), left -> -b*(1-rho)
    slope_R = np.diff(svi_total_variance([50.0, 51.0], TRUE))[0]
    assert slope_R == pytest.approx(TRUE.b * (1 + TRUE.rho), rel=1e-3)
    slope_L = np.diff(svi_total_variance([-51.0, -50.0], TRUE))[0]
    assert slope_L == pytest.approx(-TRUE.b * (1 - TRUE.rho), rel=1e-3)


def test_exact_recovery_curve_identical():
    iv_true = svi_iv(K_GRID, T, TRUE)
    params, report = fit_svi_slice(K_GRID, iv_true, T)
    assert report["rmse_iv"] < 1e-6
    # curve match on a denser grid (params may trade off, curve must not)
    kk = np.linspace(-0.5, 0.5, 200)
    np.testing.assert_allclose(svi_iv(kk, T, params), svi_iv(kk, T, TRUE), atol=5e-5)


def test_noise_robustness():
    rng = np.random.default_rng(3)
    noise = 0.004                                     # 0.4 vol pt quote noise
    iv_noisy = svi_iv(K_GRID, T, TRUE) + noise * rng.standard_normal(len(K_GRID))
    params, report = fit_svi_slice(K_GRID, iv_noisy, T)
    assert report["rmse_iv"] < 2.5 * noise            # fit absorbs, doesn't blow up
    kk = np.linspace(-0.35, 0.35, 100)
    assert np.max(np.abs(svi_iv(kk, T, params) - svi_iv(kk, T, TRUE))) < 4 * noise


def test_fit_respects_bounds():
    iv_true = svi_iv(K_GRID, T, TRUE)
    params, _ = fit_svi_slice(K_GRID, iv_true, T)
    assert params.b >= 0
    assert -1 < params.rho < 1
    assert params.sigma > 0


def test_otm_side_selection():
    df = pd.DataFrame({
        "strike": [90.0, 95.0, 105.0, 110.0, 90.0, 95.0, 105.0, 110.0],
        "option_type": ["P", "P", "P", "P", "C", "C", "C", "C"],
        "F": 100.0,
        "status": "ok",
    })
    sel = otm_side(df)
    assert len(sel) == 4
    assert ((sel["strike"] < 100) == (sel["option_type"] == "P")).all()


# --- Real data ----------------------------------------------------------------

@pytest.fixture(scope="module")
def real_slice():
    path = PROCESSED_DIR / "iv_surface.parquet"
    if not path.exists():
        pytest.skip("iv surface not built")
    surf = pd.read_parquet(path)
    ok = surf[surf["status"] == "ok"]
    date, expiry = ok.groupby(["date", "expiry"]).size().idxmax()
    return otm_side(ok[(ok["date"] == date) & (ok["expiry"] == expiry)])


# --- Day 10: all slices ---------------------------------------------------

def synth_surface_two_slices():
    """Two synthetic slices as an iv_surface-shaped frame (OTM rows only)."""
    rows = []
    for expiry, T, params in (("2023-07-21", 0.15, TRUE),
                              ("2023-09-15", 0.30, SVIParams(0.02, 0.30, -0.55, 0.02, 0.22))):
        for k in K_GRID:
            rows.append({"date": "2023-06-02", "expiry": expiry, "T": T,
                         "strike": 100 * np.exp(k), "F": 100.0,
                         "option_type": "P" if k < 0 else "C",
                         "log_moneyness": k, "iv": float(svi_iv(k, T, params)),
                         "status": "ok"})
    return pd.DataFrame(rows)


def test_fit_all_slices_synthetic():
    fits = fit_all_slices(synth_surface_two_slices())
    assert len(fits) == 2
    assert fits["fit_ok"].all()
    assert (fits["rmse_iv"] < 1e-5).all()


def test_fit_all_skips_thin_slices():
    surf = synth_surface_two_slices()
    thin = surf[surf["expiry"] == "2023-07-21"].head(3)      # < MIN_POINTS
    full = surf[surf["expiry"] == "2023-09-15"]
    fits = fit_all_slices(pd.concat([thin, full], ignore_index=True))
    assert len(fits) == 2
    assert fits.set_index("expiry")["fit_ok"].to_dict() == {
        "2023-07-21": False, "2023-09-15": True}


@pytest.fixture(scope="module")
def real_fits():
    path = PROCESSED_DIR / "iv_surface.parquet"
    if not path.exists():
        pytest.skip("iv surface not built")
    return fit_all_slices(pd.read_parquet(path))


def test_real_all_slices_fit(real_fits):
    ok = real_fits[real_fits["fit_ok"]]
    assert len(ok) >= 12                                     # 15 slices, allow few thin
    assert (ok["rmse_iv"].median() * 100) < 1.0              # median < 1 volpt
    assert (ok["rmse_iv"].max() * 100) < 3.0                 # worst < 3 volpts
    assert ok["rho"].between(-1, 1).all()
    assert (ok["b"] >= 0).all()
    assert (ok["min_w_on_grid"] >= 0).all()                  # no negative variance on quotes


def test_real_param_time_series_stable(real_fits):
    # same expiry across quote dates: ATM total variance drifts smoothly.
    # Overfit proxy: w_atm = a + b*(rho*(0-m)+sqrt(m^2+sigma^2)) should not
    # swing wildly date-to-date for the same expiry (smile itself is stable).
    ok = real_fits[real_fits["fit_ok"]].copy()
    ok["w_atm"] = ok["a"] + ok["b"] * (ok["rho"] * (0 - ok["m"])
                                       + np.sqrt(ok["m"] ** 2 + ok["sigma"] ** 2))
    ok["atm_iv"] = np.sqrt(ok["w_atm"] / ok["T"])
    for expiry, g in ok.groupby("expiry"):
        if len(g) < 2:
            continue
        assert g["atm_iv"].std() < 0.05, (expiry, g["atm_iv"].tolist())


def test_real_slice_fit_quality(real_slice):
    T_sl = float(real_slice["T"].iloc[0])
    params, report = fit_svi_slice(real_slice["log_moneyness"], real_slice["iv"], T_sl)
    assert report["n_points"] >= 10
    assert report["rmse_iv"] < 0.02                   # < 2 vol pts on raw single-name data
    assert report["min_w_on_grid"] >= 0               # no negative variance on quotes
    kk = np.linspace(real_slice["log_moneyness"].min(), real_slice["log_moneyness"].max(), 200)
    assert (svi_total_variance(kk, params) >= 0).all()
