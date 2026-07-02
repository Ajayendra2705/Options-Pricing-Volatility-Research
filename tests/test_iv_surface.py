"""
Day 8 tests: raw IV surface from cleaned chain + parity forwards.

- Synthetic full-path: chain built from a known smile -> forwards implied
  by Day 7 -> surface inverted by Day 8 recovers the smile (integration of
  Days 5+6+7+8 machinery).
- Failure classification: below-intrinsic and impossible quotes flagged,
  never silently dropped.
- Real data: high inversion success, sane IV band, call/put IV parity
  consistency near ATM (same strike, same slice).
"""

import numpy as np
import pandas as pd
import pytest

from src.greeks import black_scholes as bs
from src.surface.clean import PROCESSED_DIR
from src.surface.forwards import imply_forwards
from src.surface.iv_surface import build_iv_surface, surface_report


def synth_chain(F=185.0, r=0.05, T=0.4, strikes=None, expiry="2023-10-20"):
    """Chain with a known skewed smile; returns (chain, sigma_fn)."""
    strikes = np.asarray(strikes if strikes is not None else np.arange(150.0, 220.0, 5.0))

    def sigma_fn(K):
        k = np.log(np.asarray(K, float) / F)
        return 0.22 - 0.25 * k + 0.6 * k**2          # skew + smile

    rows = []
    for cp, name in ((1, "C"), (-1, "P")):
        mids = bs.price(F, strikes, T, sigma_fn(strikes), r, cp)
        for K, m in zip(strikes, mids):
            rows.append({"date": "2023-06-02", "expiry": expiry, "strike": float(K),
                         "option_type": name, "mid": float(m), "T": T})
    return pd.DataFrame(rows), sigma_fn


def test_full_path_smile_recovery():
    chain, sigma_fn = synth_chain()
    forwards = imply_forwards(chain, r=0.05)          # Day 7 machinery
    surf = build_iv_surface(chain, forwards)          # Day 8
    ok = surf[surf["status"] == "ok"]
    assert len(ok) == len(chain)                      # everything inverts
    np.testing.assert_allclose(ok["iv"], sigma_fn(ok["strike"]), atol=5e-4)
    # call and put at the same strike give the same IV (shared forward)
    piv = ok.pivot_table(index="strike", columns="option_type", values="iv")
    np.testing.assert_allclose(piv["C"], piv["P"], atol=5e-4)


def test_failure_classification_flagged_not_dropped():
    chain, _ = synth_chain(strikes=np.array([170.0, 185.0, 200.0]))
    forwards = imply_forwards(chain, r=0.05)
    # corrupt one quote below intrinsic and one absurdly high
    chain.loc[0, "mid"] = 0.5 * max(185.0 - chain.loc[0, "strike"], 0.0) if chain.loc[0, "strike"] < 185 else 0.001
    chain.loc[1, "mid"] = 500.0
    surf = build_iv_surface(chain, forwards)
    assert len(surf) == len(chain)                    # nothing silently dropped
    assert (surf["status"] != "ok").sum() >= 2
    assert set(surf["status"].unique()) <= {"ok", "below_intrinsic", "above_upper", "no_solution"}
    assert surf.loc[1, "status"] == "above_upper"


def test_report_structure():
    chain, _ = synth_chain()
    forwards = imply_forwards(chain, r=0.05)
    rep = surface_report(build_iv_surface(chain, forwards))
    assert rep["success_rate"] == 1.0
    assert rep["n_rows"] == len(chain)
    assert len(rep["slices"]) == 1
    assert rep["slices"][0]["n_ok"] == len(chain)
    assert rep["slices"][0]["wing_coverage_4sig"] > 0


# --- Real data ---------------------------------------------------------------

@pytest.fixture(scope="module")
def real_surface():
    chain_p = PROCESSED_DIR / "chain_clean.parquet"
    fwd_p = PROCESSED_DIR / "forwards.parquet"
    if not (chain_p.exists() and fwd_p.exists()):
        pytest.skip("processed data not built")
    return build_iv_surface(pd.read_parquet(chain_p), pd.read_parquet(fwd_p))


def test_real_success_rate(real_surface):
    rep = surface_report(real_surface)
    assert rep["success_rate"] > 0.90, rep["status_counts"]


def test_real_iv_band(real_surface):
    ok = real_surface[real_surface["status"] == "ok"]
    assert ok["iv"].between(0.03, 1.5).all()          # single-name equity vol band
    # ATM vols specifically in a tight sane band for AAPL June 2023 (~15-25%)
    atm = ok[ok["log_moneyness"].abs() < 0.02]
    assert 0.10 < atm["iv"].median() < 0.35


def test_real_parity_iv_consistency(real_surface):
    # same strike, same slice, near ATM: C and P IV should agree closely
    ok = real_surface[(real_surface["status"] == "ok")
                      & (real_surface["log_moneyness"].abs() < 0.05)]
    piv = ok.pivot_table(index=["date", "expiry", "strike"], columns="option_type",
                         values="iv").dropna()
    assert len(piv) >= 20
    diff = (piv["C"] - piv["P"]).abs()
    assert diff.median() < 0.02                       # < 2 vol pts typical
    assert diff.max() < 0.06, piv.loc[diff.idxmax()]  # worst microstructure case


def test_real_failures_are_wings(real_surface):
    bad = real_surface[real_surface["status"] != "ok"]
    if len(bad) == 0:
        return
    # failures should live away from the money (wing/EEP artifacts)
    assert bad["log_moneyness"].abs().median() > 0.05
