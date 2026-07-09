"""
Day 28 — Statistical-honesty tests.

1. nw_mean_se at lag 0 equals sqrt(gamma0 / T) (classic SE of the mean, ddof=0).
2. Positive autocorrelation inflates the HAC SE above the lag-0 value.
3. annualized_sharpe = mean/std * sqrt(ppy).
4. sharpe_with_nw t-stat == mean / nw_mean_se (HAC t of the Sharpe = HAC t of
   the mean).
5. Block bootstrap: seeded -> reproducible; ordered CI; brackets the true
   Sharpe on a large IID sample.
6. IS/OOS haircut splits chronologically and is the SR difference.
7. Real-data gate (skip if files missing): statistical_honesty lands in
   metrics.json with a negative Sharpe and finite CI.
"""

import json

import numpy as np
import pytest

from src.backtest.stats import (
    annualized_sharpe,
    block_bootstrap_sharpe,
    is_oos_haircut,
    nw_mean_se,
    sharpe_with_nw,
    TRADING_DAYS,
)
from src.backtest.reconcile import PROCESSED_DIR, RAW_DIR, RESULTS_DIR


# ── 1-2. HAC SE of the mean ──────────────────────────────────────────────────

def test_nw_mean_se_lag0():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(200) * 0.01 + 0.001
    se0 = nw_mean_se(r, 0)
    gamma0 = ((r - r.mean()) ** 2).mean()          # ddof=0
    assert se0 == pytest.approx(np.sqrt(gamma0 / r.size))


def test_nw_mean_se_autocorrelation_inflates():
    rng = np.random.default_rng(4)
    n = 400
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.7 * e[t - 1] + rng.standard_normal()
    assert nw_mean_se(e, 12) > nw_mean_se(e, 0)


# ── 3-4. Sharpe + HAC t ──────────────────────────────────────────────────────

def test_annualized_sharpe_formula():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(150) * 0.01 + 0.0008
    assert annualized_sharpe(r) == pytest.approx(
        r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def test_sharpe_nw_tstat_equals_mean_over_se():
    rng = np.random.default_rng(2)
    r = rng.standard_normal(120) * 0.01 + 0.0005
    out = sharpe_with_nw(r, n_lags=5)
    assert out["nw_tstat"] == pytest.approx(r.mean() / nw_mean_se(r, 5))


# ── 5. block bootstrap ───────────────────────────────────────────────────────

def test_block_bootstrap_reproducible_and_ordered():
    rng = np.random.default_rng(9)
    r = rng.standard_normal(200) * 0.01
    a = block_bootstrap_sharpe(r, block=10, n_boot=500, seed=7)
    b = block_bootstrap_sharpe(r, block=10, n_boot=500, seed=7)
    assert a == b                                   # seeded, deterministic
    assert a["ci_2.5"] <= a["ci_50"] <= a["ci_97.5"]


def test_block_bootstrap_brackets_true_sharpe():
    rng = np.random.default_rng(11)
    r = rng.standard_normal(1000) * 0.01 + 0.002    # positive-Sharpe IID
    true_sr = annualized_sharpe(r)
    ci = block_bootstrap_sharpe(r, block=5, n_boot=1000, seed=0)
    assert ci["ci_2.5"] <= true_sr <= ci["ci_97.5"]


# ── 6. IS/OOS haircut ────────────────────────────────────────────────────────

def test_is_oos_haircut_split():
    r = np.arange(1.0, 11.0) * 0.001                # 10 obs
    h = is_oos_haircut(r)
    assert h["n_is"] == 5 and h["n_oos"] == 5
    assert h["haircut_is_minus_oos"] == pytest.approx(
        annualized_sharpe(r[:5]) - annualized_sharpe(r[5:]))


# ── 7. real-data gate ────────────────────────────────────────────────────────

_NEEDED = [PROCESSED_DIR / "returns.parquet",
           PROCESSED_DIR / "signal.parquet",
           RAW_DIR / "aapl_ohlc.parquet",
           RAW_DIR / "aapl_ohlc_ext.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="real data files not present")
def test_run_stats_real():
    from src.backtest.stats import run_stats

    m = run_stats()
    s = m["statistical_honesty"]
    assert s["sharpe"]["sharpe_annualized"] < 0        # negative after costs
    assert np.isfinite(s["sharpe"]["nw_se"])
    assert np.isfinite(s["sharpe"]["nw_tstat"])
    for k in ("ci_2.5", "ci_50", "ci_97.5"):
        assert np.isfinite(s["bootstrap_ci_95"][k])
    assert s["bootstrap_ci_95"]["ci_2.5"] <= s["bootstrap_ci_95"]["ci_97.5"]
    assert s["deflated_sharpe"]["computed"] is False    # deferred to v2
    # persisted
    disk = json.loads((RESULTS_DIR / "metrics.json").read_text())
    assert "statistical_honesty" in disk
