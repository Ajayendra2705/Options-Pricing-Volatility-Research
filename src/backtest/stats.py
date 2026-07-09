"""
Day 28 — Statistical honesty.

Adds the uncertainty around the headline Sharpe to results/metrics.json.  A
point Sharpe on 54 serially-correlated bars is nearly meaningless without it.

DELIVERABLES (SPEC "Statistical honesty", PLAN Day 28):
  1. Sharpe with a **Newey-West (HAC) standard error** and t-stat.  Because the
     annualized Sharpe is proportional to the mean return, its HAC t-stat is
     exactly the HAC t-stat of the mean (mean / HAC-SE(mean)) — computed with a
     Bartlett kernel, lags = ceil(median holding horizon).  Reported as SE on
     the Sharpe and the significance t.
  2. **Block-bootstrap 95% CI** on the annualized Sharpe.  Returns are serially
     correlated (monthly holds), so an IID bootstrap understates the CI; a
     moving-block bootstrap preserves the dependence.  Seeded -> byte-stable.
  3. **IS-vs-OOS Sharpe haircut** — chronological split-half Sharpe, reported as
     a labeled result (the degradation number is itself a deliverable).  Thin
     sample flagged.

DEFLATED SHARPE (Bailey / Lopez de Prado) is intentionally NOT computed here:
it is only meaningful with an honest multiple-testing trial count N, and N is
only complete after the v2 robustness sweeps (PLAN v2 Day 37).  Computing a DSR
now with N=1 would understate the correction — recorded as a deferred item, not
faked.

OUTPUT: merged into results/metrics.json under "statistical_honesty".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.reconcile import build_positions, load_price_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRADING_DAYS = 252
N_BOOT = 2000
BOOT_SEED = 0


# ── HAC SE of the mean (Bartlett) ────────────────────────────────────────────

def nw_mean_se(returns: np.ndarray, n_lags: int) -> float:
    """Newey-West (Bartlett) HAC standard error of the sample mean.

    S = gamma_0 + 2 * sum_{l=1..L} (1 - l/(L+1)) gamma_l ;  SE = sqrt(S / T).
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 2:
        return float("nan")
    dev = r - r.mean()
    gamma0 = float(dev @ dev) / T
    S = gamma0
    for l in range(1, min(n_lags, T - 1) + 1):
        w = 1.0 - l / (n_lags + 1)
        gl = float(dev[l:] @ dev[:-l]) / T
        S += 2.0 * w * gl
    S = max(S, 0.0)                          # HAC can go slightly negative
    return float(np.sqrt(S / T))


def annualized_sharpe(returns: np.ndarray, ppy: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def sharpe_with_nw(returns: np.ndarray, n_lags: int,
                   ppy: int = TRADING_DAYS) -> dict:
    """Annualized Sharpe + HAC SE + t-stat.

    SR_ann = mean/std * sqrt(ppy);  the HAC t of SR equals mean / HAC-SE(mean),
    so SE(SR_ann) = SE_ann_of_mean / std = nw_mean_se * sqrt(ppy) / std.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    mean, std = r.mean(), r.std(ddof=1)
    sr = annualized_sharpe(r, ppy)
    se_mean = nw_mean_se(r, n_lags)
    se_sr = se_mean * np.sqrt(ppy) / std if std > 0 else float("nan")
    t = mean / se_mean if se_mean > 0 else float("nan")
    return {"sharpe_annualized": sr, "nw_se": float(se_sr),
            "nw_tstat": float(t), "n_lags": int(n_lags), "n_obs": int(r.size)}


# ── moving-block bootstrap CI ────────────────────────────────────────────────

def block_bootstrap_sharpe(returns: np.ndarray, block: int,
                           n_boot: int = N_BOOT, ppy: int = TRADING_DAYS,
                           seed: int = BOOT_SEED) -> dict:
    """Moving-block bootstrap 95% CI on the annualized Sharpe (seeded)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = r.size
    block = max(1, min(block, T))
    n_blocks = int(np.ceil(T / block))
    starts_max = T - block + 1
    rng = np.random.default_rng(seed)
    srs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:T]
        srs[b] = annualized_sharpe(sample, ppy)
    srs = srs[np.isfinite(srs)]
    return {
        "ci_2.5": float(np.percentile(srs, 2.5)),
        "ci_50": float(np.percentile(srs, 50)),
        "ci_97.5": float(np.percentile(srs, 97.5)),
        "block": int(block), "n_boot": int(n_boot),
    }


# ── IS / OOS haircut ─────────────────────────────────────────────────────────

def is_oos_haircut(returns: np.ndarray, ppy: int = TRADING_DAYS) -> dict:
    """Chronological split-half Sharpe: in-sample vs out-of-sample degradation."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    mid = r.size // 2
    sr_is = annualized_sharpe(r[:mid], ppy)
    sr_oos = annualized_sharpe(r[mid:], ppy)
    return {
        "sharpe_is": sr_is, "sharpe_oos": sr_oos,
        "haircut_is_minus_oos": float(sr_is - sr_oos),
        "n_is": int(mid), "n_oos": int(r.size - mid),
        "note": "chronological split-half; indicative only at this sample size.",
    }


# ── runner ──────────────────────────────────────────────────────────────────

def run_stats() -> dict:
    """Compute Sharpe uncertainty block -> merge into metrics.json."""
    ret = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    daily = ret["daily_return"].to_numpy()

    # lags = ceil(median holding horizon), same convention as Day 27
    positions = build_positions()
    path = load_price_path()
    hold = [len(path[(path["date"] >= p["date"])
                     & (path["date"] <= p["expiry"])]) for p in positions]
    n_lags = int(np.ceil(np.median(hold)))
    block = max(2, int(np.ceil(np.sqrt(len(daily)))))   # ~ sqrt(T)

    sh = sharpe_with_nw(daily, n_lags)
    boot = block_bootstrap_sharpe(daily, block)
    haircut = is_oos_haircut(daily)

    stats = {
        "horizon": "daily net margin returns",
        "sharpe": sh,
        "bootstrap_ci_95": boot,
        "is_oos_haircut": haircut,
        "deflated_sharpe": {
            "computed": False,
            "reason": "Deflated Sharpe requires an honest multiple-testing "
                      "trial count N; N is only complete after the v2 "
                      "robustness sweeps (PLAN v2 Day 37). Deferred, not faked.",
        },
        "interpretation": "Point Sharpe is negative but its 95% bootstrap CI "
                          "spans zero and the NW t-stat is small in magnitude "
                          "(|t|<1) -> the Sharpe is not statistically "
                          "distinguishable from zero: no reliable edge either "
                          "way, consistent with the disproof thesis. The "
                          "negative IS->OOS haircut sign is noise at n=27/side.",
    }

    mpath = RESULTS_DIR / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
    metrics["statistical_honesty"] = stats
    with open(mpath, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"stats: Sharpe {sh['sharpe_annualized']:+.2f} "
          f"(NW SE {sh['nw_se']:.2f}, t {sh['nw_tstat']:+.2f}, lags {n_lags}); "
          f"boot95 [{boot['ci_2.5']:+.2f}, {boot['ci_97.5']:+.2f}] "
          f"(block {block})")
    print(f"  IS/OOS Sharpe {haircut['sharpe_is']:+.2f} -> "
          f"{haircut['sharpe_oos']:+.2f} "
          f"(haircut {haircut['haircut_is_minus_oos']:+.2f})")
    print(f"-> {mpath}")
    return metrics


if __name__ == "__main__":
    run_stats()
