"""
Day 12 tests: butterfly-constrained SVI refit.

- Market data generated FROM Vogt params (arbitrable): unconstrained fit
  reproduces the violation; constrained refit is arb-free with acceptable
  RMSE cost and stays close to the quotes.
- Benign market: constraint inactive, identical to unconstrained.
- Real surface: every fitted slice arb-free post-refit; violation-log
  fields populated.
"""

import numpy as np
import pandas as pd
import pytest

from src.surface.clean import PROCESSED_DIR
from src.surface.no_arb import (
    check_butterfly,
    fit_svi_constrained,
    refit_all_constrained,
)
from src.surface.svi import SVIParams, fit_svi_slice, svi_iv

VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
BENIGN = SVIParams(a=0.015, b=0.35, rho=-0.65, m=0.05, sigma=0.25)
T_VOGT = 1.0     # Vogt params are quoted as a total-variance slice at T=1
K_GRID = np.linspace(-0.9, 0.9, 31)


def test_vogt_market_unconstrained_reproduces_violation():
    iv = svi_iv(K_GRID, T_VOGT, VOGT)
    params, _ = fit_svi_slice(K_GRID, iv, T_VOGT)
    assert not check_butterfly(params)["arb_free"]     # faithful fit inherits the arb


def test_vogt_market_constrained_is_arb_free():
    iv = svi_iv(K_GRID, T_VOGT, VOGT)
    params, rep = fit_svi_constrained(K_GRID, iv, T_VOGT)
    assert rep["constraint_active"]
    assert not rep["pre_arb_free"]
    assert rep["arb_free"], rep
    assert rep["min_g"] >= 0
    # constrained curve still close to the (arbitrable) quotes:
    # Gatheral-Jacquier report ~0.3 volpt distance for this case
    assert rep["rmse_iv"] < 0.006
    # and the fit cost exceeds the unconstrained one (constraint binds)
    assert rep["rmse_iv"] >= rep["rmse_iv_unconstrained"]


def test_benign_market_constraint_inactive():
    iv = svi_iv(K_GRID, 0.35, BENIGN)
    params, rep = fit_svi_constrained(K_GRID, iv, 0.35)
    assert not rep["constraint_active"]
    assert rep["method"] == "unconstrained"
    assert rep["arb_free"]
    assert rep["rmse_iv"] == rep["rmse_iv_unconstrained"]
    assert rep["rmse_iv"] < 1e-6


# --- Real surface -------------------------------------------------------------

@pytest.fixture(scope="module")
def real_refit():
    path = PROCESSED_DIR / "iv_surface.parquet"
    if not path.exists():
        pytest.skip("iv surface not built")
    return refit_all_constrained(pd.read_parquet(path))


def test_real_all_arb_free_post_refit(real_refit):
    ok = real_refit[real_refit["fit_ok"]]
    assert len(ok) >= 12
    assert ok["arb_free"].all()
    assert (ok["min_g"] >= 0).all()


def test_real_rmse_not_degraded(real_refit):
    ok = real_refit[real_refit["fit_ok"]]
    # this window's raw fits were already arb-free -> constraint unbinding,
    # constrained RMSE must equal unconstrained
    unbinding = ok[~ok["constraint_active"].astype(bool)]
    assert (unbinding["rmse_iv"] == unbinding["rmse_iv_unconstrained"]).all()
    assert (ok["rmse_iv"].median() * 100) < 1.0


def test_real_log_fields_complete(real_refit):
    ok = real_refit[real_refit["fit_ok"]]
    for col in ("pre_arb_free", "arb_free", "method", "min_g",
                "rmse_iv", "rmse_iv_unconstrained"):
        assert col in ok.columns
        assert ok[col].notna().all()
