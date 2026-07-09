"""
Day 27 — Event PnL table + alpha isolation tests.

1. OLS point estimates match np.linalg.lstsq exactly.
2. Perfect linear fit -> zero residual -> zero SE, R^2 = 1, exact alpha/beta.
3. Newey-West lag-0 SE equals the White (HC0) sandwich SE, computed
   independently in the test.
4. Positive residual autocorrelation inflates the NW SE (L>0) above the
   White (L=0) SE — the whole point of HAC.
5. VRP factor: a Series over the trade window, finite after the first bar.
6. Event table: flags the >3% move, carries book PnL + margin change.
7. Real-data gate (skip if files missing): regression + event table land in
   metrics.json; alpha t-stat small (no residual edge), R^2 in [0,1],
   2023-08-04 present and flagged as an event.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.backtest.alpha import (
    build_event_table,
    build_vrp_factor,
    newey_west_ols,
)
from src.backtest.reconcile import (PROCESSED_DIR, RAW_DIR, RESULTS_DIR,
                                    build_positions, load_price_path)


# ── 1-2. OLS correctness ─────────────────────────────────────────────────────

def test_ols_matches_lstsq():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100)
    y = 0.5 + 2.0 * x + rng.standard_normal(100) * 0.1
    reg = newey_west_ols(y, x, n_lags=0)
    X = np.column_stack([np.ones_like(x), x])
    beta_ref, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert reg["alpha"] == pytest.approx(beta_ref[0])
    assert reg["beta"] == pytest.approx(beta_ref[1])


def test_perfect_fit_zero_se():
    x = np.linspace(-1, 1, 40)
    y = 1.0 - 3.0 * x                    # exact line, no noise
    reg = newey_west_ols(y, x, n_lags=3)
    assert reg["alpha"] == pytest.approx(1.0)
    assert reg["beta"] == pytest.approx(-3.0)
    assert reg["alpha_se"] == pytest.approx(0.0, abs=1e-9)
    assert reg["beta_se"] == pytest.approx(0.0, abs=1e-9)
    assert reg["r_squared"] == pytest.approx(1.0)


# ── 3. NW lag-0 == White HC0 ─────────────────────────────────────────────────

def test_nw_lag0_equals_white():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(80)
    y = 0.2 + 1.5 * x + rng.standard_normal(80) * (0.5 + np.abs(x))  # heterosk.
    reg = newey_west_ols(y, x, n_lags=0)

    # independent White HC0 sandwich
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    S = (X * u[:, None]).T @ (X * u[:, None])
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    assert reg["alpha_se"] == pytest.approx(se[0])
    assert reg["beta_se"] == pytest.approx(se[1])


# ── 4. autocorrelation inflates HAC SE ───────────────────────────────────────

def test_autocorrelation_inflates_se():
    rng = np.random.default_rng(5)
    n = 300
    x = rng.standard_normal(n)
    # AR(1) residuals (strong positive autocorrelation)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.standard_normal()
    y = 0.0 + 1.0 * x + e
    se0 = newey_west_ols(y, x, n_lags=0)["alpha_se"]
    seL = newey_west_ols(y, x, n_lags=10)["alpha_se"]
    assert seL > se0            # HAC widens the SE under serial correlation


# ── 6. event table ───────────────────────────────────────────────────────────

def test_event_table_flags_big_move():
    dates = pd.bdate_range("2023-06-02", periods=6)
    ret = pd.DataFrame({
        "date": dates,
        "daily_return": [0.0, 0.001, -0.002, 0.02, -0.001, 0.0],
        "book_margin": [1000, 1100, 1050, 900, 950, 1000.0],
        "net_equity": [0, 1, -1, 5, 4, 4.0],
    })
    # underlying: a +4% jump on the 4th date
    close = np.array([180, 181, 180, 187.2, 187, 187.0])
    path = pd.DataFrame({"date": dates, "close": close})
    tbl = build_event_table(ret, path)
    ev = [e for e in tbl if e["is_event"]]
    assert len(ev) == 1
    assert ev[0]["date"] == "2023-06-07"          # the +4% day (4th bday)
    assert ev[0]["underlying_return"] > 0.03
    assert "margin_change" in ev[0]


# ── 5 & 7. real-data gate ────────────────────────────────────────────────────

_NEEDED = [PROCESSED_DIR / "returns.parquet",
           PROCESSED_DIR / "signal.parquet",
           RESULTS_DIR / "returns_summary.json",
           RAW_DIR / "aapl_ohlc.parquet",
           RAW_DIR / "aapl_ohlc_ext.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="real data files not present")
def test_vrp_factor_real():
    positions = build_positions()
    path = load_price_path()
    fac = build_vrp_factor(positions, path, capital_base=27000.0)
    assert isinstance(fac, pd.Series)
    assert fac.iloc[1:].notna().all()             # finite after bar-0 diff
    assert len(fac) > 10


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="real data files not present")
def test_run_alpha_real():
    from src.backtest.metrics import run_metrics
    from src.backtest.alpha import run_alpha

    run_metrics()
    m = run_alpha()

    reg = m["alpha_regression"]
    assert 0.0 <= reg["r_squared"] <= 1.0
    assert reg["n_lags"] >= 1
    assert np.isfinite(reg["alpha_t"]) and np.isfinite(reg["beta_t"])
    # no residual edge: |alpha t| small (insignificant)
    assert abs(reg["alpha_t"]) < 2.0

    days = m["event_table"]["days"]
    ev = [e for e in days if e["is_event"]]
    assert any(e["date"] == "2023-08-04" for e in ev)   # the earnings gap

    # persisted to metrics.json
    disk = json.loads((RESULTS_DIR / "metrics.json").read_text())
    assert "alpha_regression" in disk
    assert "event_table" in disk
