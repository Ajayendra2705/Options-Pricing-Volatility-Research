"""
Day 26 — Return-distribution statistics tests.

1. max_drawdown on a crafted cumulative series matches the known trough.
2. CVaR = mean of the worst ceil(alpha*n) returns, exact on a simple array.
3. Sortino downside-deviation formula matches a manual recompute.
4. distribution_stats: mean/std/skew/kurtosis match numpy/scipy directly.
5. Sharpe annualization: mean/std * sqrt(ppy).
6. Calmar = annualized return / |maxDD|.
7. Real-data gate (skip if files missing): metrics.json has daily/weekly/
   per_trade blocks, finite, net Sharpe negative (returns negative after costs).
"""

import json
import numpy as np
import pytest
from scipy import stats as sps

from src.backtest.metrics import (
    cvar,
    distribution_stats,
    max_drawdown,
    merge_metrics,
    METRICS_KEY_ORDER,
    sortino,
    TRADING_DAYS,
)
from src.backtest.reconcile import PROCESSED_DIR, RESULTS_DIR, RAW_DIR


# ── 1. max drawdown ──────────────────────────────────────────────────────────

def test_max_drawdown_known():
    # cumulative: peak 5 at idx2, trough 1 at idx4 -> DD = 4
    cum = np.array([0.0, 3.0, 5.0, 2.0, 1.0, 4.0])
    assert max_drawdown(cum) == pytest.approx(4.0)


def test_max_drawdown_monotone_up_is_zero():
    assert max_drawdown(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(0.0)


# ── 2. CVaR ──────────────────────────────────────────────────────────────────

def test_cvar_worst_fraction():
    r = np.arange(1.0, 101.0)          # 1..100, n=100
    # alpha=0.05 -> k=5 -> worst 5 are 1,2,3,4,5 -> mean 3
    assert cvar(r, 0.05) == pytest.approx(3.0)


def test_cvar_small_sample_takes_worst_one():
    r = np.array([-0.05, 0.01, 0.02, 0.03])   # n=4, ceil(0.05*4)=1
    assert cvar(r, 0.05) == pytest.approx(-0.05)


# ── 3. Sortino ───────────────────────────────────────────────────────────────

def test_sortino_manual():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    ppy = 252
    downside = np.minimum(r, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    expect = r.mean() / dd * np.sqrt(ppy)
    assert sortino(r, ppy) == pytest.approx(expect)


# ── 4. shape stats match scipy ───────────────────────────────────────────────

def test_distribution_stats_match_reference():
    rng = np.random.default_rng(7)
    r = rng.standard_normal(200) * 0.01 + 0.0005
    s = distribution_stats(r, TRADING_DAYS, annualize=True)
    assert s["mean"] == pytest.approx(r.mean())
    assert s["std"] == pytest.approx(r.std(ddof=1))
    assert s["skew"] == pytest.approx(sps.skew(r, bias=False))
    assert s["excess_kurtosis"] == pytest.approx(
        sps.kurtosis(r, fisher=True, bias=False))
    assert s["win_rate"] == pytest.approx((r > 0).mean())


# ── 5. Sharpe annualization ──────────────────────────────────────────────────

def test_sharpe_annualized():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(300) * 0.01 + 0.001
    s = distribution_stats(r, TRADING_DAYS, annualize=True)
    expect = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert s["sharpe"] == pytest.approx(expect)


def test_sharpe_non_annualized_per_trade():
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    s = distribution_stats(r, None, annualize=False)
    assert s["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1))


# ── 6. Calmar ────────────────────────────────────────────────────────────────

def test_calmar_ann_return_over_maxdd():
    r = np.array([0.01, -0.03, 0.02, -0.01, 0.02, 0.01])
    s = distribution_stats(r, TRADING_DAYS, annualize=True)
    mdd = max_drawdown(np.cumsum(r))
    expect = (r.mean() * TRADING_DAYS) / mdd
    assert s["calmar"] == pytest.approx(expect)


# ── 7. real-data gate ────────────────────────────────────────────────────────

_NEEDED = [PROCESSED_DIR / "returns.parquet",
           RESULTS_DIR / "returns_summary.json",
           RESULTS_DIR / "costs_summary.json",
           RAW_DIR / "aapl_ohlc.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                    reason="real data files not present")
def test_real_data_metrics():
    from src.backtest.metrics import run_metrics

    m = run_metrics()
    for h in ("daily", "weekly", "per_trade"):
        assert h in m["horizons"]
        blk = m["horizons"][h]
        assert blk["n"] > 0
        assert np.isfinite(blk["mean"])
        for k in ("skew", "cvar_5pct", "max_drawdown"):
            assert k in blk

    # 10 trades
    assert m["horizons"]["per_trade"]["n"] == 10
    # net returns are negative after costs -> negative daily Sharpe
    assert m["net_return_on_capital"] < 0
    assert m["horizons"]["daily"]["sharpe"] < 0
    # denominator carried through from Day 25
    assert "Reg-T margin" in m["denominator"]


# ── metrics.json is shared: merging must not drop the other days' blocks ─────

def test_merge_metrics_preserves_other_days_blocks(tmp_path):
    """Regression: run_metrics used to OVERWRITE metrics.json, silently dropping
    the Day-27 (alpha_regression/event_table) and Day-28 (statistical_honesty)
    blocks whenever it ran on its own."""
    merge_metrics({"alpha_regression": {"beta": -1.59},
                   "statistical_honesty": {"sharpe": {"nw_tstat": -0.86}}},
                  results_dir=tmp_path)
    merged = merge_metrics({"horizons": {"daily": {"sharpe": -1.7}}},
                           results_dir=tmp_path)

    assert merged["alpha_regression"]["beta"] == -1.59            # not dropped
    assert merged["statistical_honesty"]["sharpe"]["nw_tstat"] == -0.86
    assert merged["horizons"]["daily"]["sharpe"] == -1.7          # own key fresh
    assert json.loads((tmp_path / "metrics.json").read_text()) == merged


def test_merge_metrics_key_order_is_canonical_regardless_of_call_order(tmp_path):
    """Byte-stability (Day-30 gate): the file's key order must depend only on
    METRICS_KEY_ORDER, never on which runner wrote first."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    blocks = {"statistical_honesty": {"x": 1}, "horizons": {"y": 2},
              "alpha_regression": {"z": 3}}

    for k, v in blocks.items():                      # stats -> horizons -> alpha
        merge_metrics({k: v}, results_dir=a)
    for k in reversed(list(blocks)):                 # reverse order
        merge_metrics({k: blocks[k]}, results_dir=b)

    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()
    keys = list(json.loads((a / "metrics.json").read_text()))
    assert keys == [k for k in METRICS_KEY_ORDER if k in blocks]
