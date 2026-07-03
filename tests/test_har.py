"""
Day 17 tests: HAR-RV forecast.

- Daily variance proxy recovers sigma^2 on fine-step GBM.
- Constant-vol GBM: forecast level unbiased (half-variance correction works).
- Persistent stochastic vol (log-vol AR(1)): HAR finds the predictability —
  decent in-sample R^2 and out-of-sample correlation with realized.
- OLS coefficients match an independent normal-equations recompute.
- No lookahead: expanding forecasts through t invariant to future bars.
- Structure: warm-up NaNs, target NaN in the last h rows.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.har import (
    HORIZON,
    MONTH,
    daily_variance_proxy,
    expanding_forecast,
    fit_har,
    har_dataset,
)
from tests.test_realized_vol import gbm_ohlc

TRADING_DAYS = 252


def sv_ohlc(n_days=900, mean_vol=0.25, phi=0.97, eta=0.10, steps=78, seed=7):
    """OHLC from a log-vol AR(1) stochastic-vol path; returns (df, true vols)."""
    rng = np.random.default_rng(seed)
    lnsig = np.empty(n_days)
    lnsig[0] = np.log(mean_vol)
    eps = rng.standard_normal(n_days)
    for t in range(1, n_days):
        lnsig[t] = (1 - phi) * np.log(mean_vol) + phi * lnsig[t - 1] + eta * eps[t]
    sig = np.exp(lnsig)
    dt = 1.0 / (TRADING_DAYS * steps)
    z = rng.standard_normal((n_days, steps))
    incr = -0.5 * sig[:, None] ** 2 * dt + sig[:, None] * np.sqrt(dt) * z
    logp = np.cumsum(incr.reshape(-1)).reshape(n_days, steps) + np.log(100.0)
    p = np.exp(logp)
    df = pd.DataFrame({
        "date": pd.bdate_range("2015-01-01", periods=n_days),
        "open": p[:, 0], "high": p.max(axis=1),
        "low": p.min(axis=1), "close": p[:, -1],
    })
    return df, sig


@pytest.fixture(scope="module")
def gbm():
    return gbm_ohlc()


@pytest.fixture(scope="module")
def sv():
    return sv_ohlc()


def test_daily_proxy_recovers_sigma2(gbm):
    v = daily_variance_proxy(gbm).dropna()
    # Rogers-Satchell has a known DOWNWARD discrete-monitoring bias, O(1/sqrt(m))
    # with m=390 steps/day it lands ~10% low on variance. Assert one-sided:
    # below sigma^2 (plus noise) but not by more than 20%.
    assert 0.80 * 0.30 ** 2 < v.mean() < 1.02 * 0.30 ** 2


def test_constant_vol_forecast_unbiased(gbm):
    ds = har_dataset(gbm)
    fit = fit_har(ds)
    # constant sigma: nothing to predict, but the LEVEL must match the proxy's
    # own scale (half-variance correction working); proxy bias cancels here
    v = daily_variance_proxy(gbm).dropna()
    assert fit["fitted"].dropna().mean() == pytest.approx(np.sqrt(v.mean()), rel=0.03)
    assert 0.25 < fit["fitted"].dropna().mean() < 0.32   # absolute sanity band


def test_har_finds_persistence(sv):
    df, _ = sv
    fit = fit_har(har_dataset(df))
    # observed 0.504 on this seed; 0.45 leaves headroom against numerical drift
    assert fit["r2"] > 0.45


def test_oos_forecast_tracks_realized(sv):
    df, _ = sv
    ds = har_dataset(df)
    oos = expanding_forecast(ds)
    both = pd.concat([oos, ds["sig_fwd"]], axis=1).dropna()
    assert len(both) > 300
    corr = np.corrcoef(both["har_oos"], both["sig_fwd"])[0, 1]
    assert corr > 0.5


def test_ols_matches_normal_equations(sv):
    df, _ = sv
    ds = har_dataset(df)
    fit = fit_har(ds)
    X = np.column_stack([
        np.ones(len(ds)),
        np.log(ds["sig_d"]), np.log(ds["sig_w"]), np.log(ds["sig_m"]),
    ])
    y = np.log(ds["sig_fwd"].to_numpy())
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    beta = np.linalg.solve(X[ok].T @ X[ok], X[ok].T @ y[ok])
    np.testing.assert_allclose(fit["beta"], beta, rtol=1e-8)


def test_no_lookahead(sv):
    df, _ = sv
    full = expanding_forecast(har_dataset(df))
    t = 700
    corrupted = df.copy()
    corrupted.loc[t + 1:, ["open", "high", "low", "close"]] *= 7.0
    corrupted["high"] = corrupted[["open", "high", "close"]].max(axis=1)
    corrupted["low"] = corrupted[["open", "low", "close"]].min(axis=1)
    part = expanding_forecast(har_dataset(corrupted))
    pd.testing.assert_series_equal(full.iloc[: t + 1], part.iloc[: t + 1])


def test_structure(gbm):
    ds = har_dataset(gbm)
    # sig_m needs MONTH bars of v; v starts at row 1 (overnight needs C_{t-1})
    assert ds["sig_m"].iloc[: MONTH].isna().all()
    assert ds["sig_m"].iloc[MONTH:].notna().all()
    # target undefined once the h-bar future window runs off the sample
    assert ds["sig_fwd"].iloc[-HORIZON:].isna().all()
    assert ds["sig_fwd"].iloc[1:-HORIZON].notna().all()


def test_fit_needs_enough_rows(gbm):
    with pytest.raises(ValueError, match="complete rows"):
        fit_har(har_dataset(gbm.head(30)))
