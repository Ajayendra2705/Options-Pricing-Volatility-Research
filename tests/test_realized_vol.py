"""
Day 16 tests: trailing realized-vol estimators.

- Synthetic GBM with intraday fine steps -> daily OHLC: Yang-Zhang and
  close-to-close both recover the known sigma; YZ has lower sampling
  variance (its raison d'etre); both are drift-robust.
- Formula check: YZ recomputed inline (independent code path) matches.
- No lookahead: estimates through t are invariant to editing bars after t.
- NaN structure: exactly the warm-up prefix is NaN.
- Real data (if downloaded): sane June-2023 levels, no NaNs after warm-up.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.realized_vol import (
    RAW_DIR,
    close_to_close_vol,
    realized_vol_table,
    yang_zhang_vol,
)

TRADING_DAYS = 252


def gbm_ohlc(n_days=1500, sigma=0.30, mu=0.0, steps=390, seed=11):
    """Daily OHLC sampled from a fine-step GBM (steps per day)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / (TRADING_DAYS * steps)
    z = rng.standard_normal((n_days, steps))
    incr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    logp = np.cumsum(incr.reshape(-1)).reshape(n_days, steps) + np.log(100.0)
    p = np.exp(logp)
    return pd.DataFrame({
        "date": pd.bdate_range("2015-01-01", periods=n_days),
        "open": p[:, 0], "high": p.max(axis=1),
        "low": p.min(axis=1), "close": p[:, -1],
    })


@pytest.fixture(scope="module")
def gbm():
    return gbm_ohlc()


def test_yz_recovers_known_sigma(gbm):
    yz = yang_zhang_vol(gbm, window=63).dropna()
    # mean estimate over ~1400 daily estimates; discrete-monitoring bias on
    # H/L shrinks RS slightly -> allow 5%
    assert yz.mean() == pytest.approx(0.30, rel=0.05)


def test_cc_recovers_known_sigma(gbm):
    cc = close_to_close_vol(gbm, window=63).dropna()
    assert cc.mean() == pytest.approx(0.30, rel=0.05)


def test_yz_more_efficient_than_cc(gbm):
    yz = yang_zhang_vol(gbm, window=21).dropna()
    cc = close_to_close_vol(gbm, window=21).dropna()
    # Yang-Zhang's variance should be well below close-to-close's
    assert yz.std() < 0.75 * cc.std()


def test_drift_robustness():
    flat = yang_zhang_vol(gbm_ohlc(n_days=800, mu=0.0, seed=5), window=63).dropna()
    drift = yang_zhang_vol(gbm_ohlc(n_days=800, mu=0.8, seed=5), window=63).dropna()
    # same shocks, huge drift: YZ estimate barely moves
    assert drift.mean() == pytest.approx(flat.mean(), rel=0.02)


def test_yz_formula_matches_inline_recompute(gbm):
    n = 21
    got = yang_zhang_vol(gbm, window=n, annualize=False)
    df = gbm
    o = np.log(df["open"].to_numpy() / np.roll(df["close"].to_numpy(), 1))
    c = np.log(df["close"].to_numpy() / df["open"].to_numpy())
    rs = (np.log(df["high"] / df["open"]) * np.log(df["high"] / df["close"])
          + np.log(df["low"] / df["open"]) * np.log(df["low"] / df["close"])).to_numpy()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    for t in (100, 500, 1400):
        sl = slice(t - n + 1, t + 1)
        var = (np.var(o[sl], ddof=1) + k * np.var(c[sl], ddof=1)
               + (1 - k) * np.mean(rs[sl]))
        assert got.iloc[t] == pytest.approx(np.sqrt(var), rel=1e-12)


def test_no_lookahead(gbm):
    full = yang_zhang_vol(gbm, window=21)
    t = 700
    corrupted = gbm.copy()
    corrupted.loc[t + 1:, ["open", "high", "low", "close"]] *= 7.0   # nuke the future
    corrupted["high"] = corrupted[["open", "high", "close"]].max(axis=1)
    part = yang_zhang_vol(corrupted, window=21)
    pd.testing.assert_series_equal(full.iloc[: t + 1], part.iloc[: t + 1])


def test_nan_warmup_structure(gbm):
    w = 21
    yz = yang_zhang_vol(gbm, window=w)
    assert yz.iloc[:w].isna().all()          # first w rows lack a full window
    assert yz.iloc[w:].notna().all()


def test_validation_errors():
    df = gbm_ohlc(n_days=50)
    with pytest.raises(ValueError, match="missing column"):
        yang_zhang_vol(df.drop(columns=["high"]))
    dup = pd.concat([df, df.iloc[[10]]])
    with pytest.raises(ValueError, match="duplicate dates"):
        yang_zhang_vol(dup)


# --- Real data ----------------------------------------------------------------

@pytest.fixture(scope="module")
def real_tab():
    path = RAW_DIR / "aapl_ohlc.parquet"
    if not path.exists():
        pytest.skip("OHLC not downloaded")
    return realized_vol_table(pd.read_parquet(path))


def test_real_levels_sane(real_tab):
    tab = real_tab.set_index("date")
    june = tab.loc["2023-06-01":"2023-06-30", "yz_21"]
    assert june.notna().all()
    # AAPL June 2023: calm regime, ATM IV ~ 20%; RV plausibly 10-30%
    assert 0.05 < june.min() and june.max() < 0.40


def test_real_no_gaps_after_warmup(real_tab):
    tab = real_tab
    for col in ("yz_10", "yz_21", "cc_21"):
        s = tab[col]
        first = s.first_valid_index()
        assert s.loc[first:].notna().all(), col
