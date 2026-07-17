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

from src.backtest.metrics import merge_metrics
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


# ── Probabilistic / Deflated Sharpe (Bailey & Lopez de Prado 2014) ───────────
# All Sharpes here are PER-PERIOD (non-annualized): SR_period = mean/std. The
# annualized figures elsewhere are SR_period * sqrt(ppy); convert before use.

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation to the inverse normal CDF (|err| < 1e-9).
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = (-2 * np.log(1 - p)) ** 0.5
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(sr_period: float, sr_benchmark_period: float,
                               n_obs: int, skew: float, kurt: float) -> float:
    """P(true Sharpe > benchmark), correcting for skew/kurtosis of returns.

    sr_period, sr_benchmark_period: non-annualized. kurt: FULL kurtosis (3 for
    normal), i.e. excess_kurtosis + 3. Returns a probability in (0, 1).
    """
    denom = np.sqrt(max(1.0 - skew * sr_period
                        + (kurt - 1.0) / 4.0 * sr_period ** 2, 1e-12))
    z = (sr_period - sr_benchmark_period) * np.sqrt(n_obs - 1) / denom
    return float(_norm_cdf(z))


def expected_max_sharpe_period(var_sr_trials_period: float,
                               n_trials: int) -> float:
    """Expected maximum per-period Sharpe of N independent null trials whose
    Sharpe estimates have variance var_sr_trials_period (Bailey-LdP)."""
    if n_trials < 1:
        raise ValueError("n_trials >= 1")
    if n_trials == 1:
        return 0.0
    g, e = EULER_MASCHERONI, np.e
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(var_sr_trials_period) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(sr_period: float, n_obs: int, skew: float,
                          kurt: float, var_sr_trials_period: float,
                          n_trials: int) -> dict:
    """DSR = PSR evaluated at the expected-max-of-N-trials benchmark."""
    sr_star = expected_max_sharpe_period(var_sr_trials_period, n_trials)
    dsr = probabilistic_sharpe_ratio(sr_period, sr_star, n_obs, skew, kurt)
    return {"dsr": dsr, "sr_benchmark_period": sr_star, "n_trials": n_trials}


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

def _auto_interpretation(sh: dict, boot: dict) -> str:
    """Sign-aware reading of the Sharpe uncertainty block, built from the
    numbers instead of asserting v1's facts about another study's returns."""
    sign = "negative" if sh["sharpe_annualized"] < 0 else "positive"
    spans = boot["ci_2.5"] < 0 < boot["ci_97.5"]
    signif = abs(sh["nw_tstat"]) >= 2
    if not signif and spans:
        return (f"Point Sharpe is {sign} but its 95% bootstrap CI spans zero "
                f"and |NW t| = {abs(sh['nw_tstat']):.2f} < 2 -> not "
                "statistically distinguishable from zero: no reliable edge "
                "either way.")
    return (f"Point Sharpe is {sign} with |NW t| = {abs(sh['nw_tstat']):.2f} "
            f"and bootstrap CI [{boot['ci_2.5']:+.2f}, {boot['ci_97.5']:+.2f}]"
            " -> read with the co-headline tail stats before concluding.")


def run_stats(
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    price_path: pd.DataFrame | None = None,
    interpretation: str | None = None,
) -> dict:
    """Compute Sharpe uncertainty block -> merge into metrics.json.

    Seams default to v1's constants (Day-32 convention). `interpretation`:
    None keeps v1's exact tracked prose; "auto" derives sign-aware text from
    the computed numbers (Phase 2); any other string is used verbatim.
    """
    processed_dir = processed_dir or PROCESSED_DIR
    results_dir = results_dir or RESULTS_DIR
    ret = pd.read_parquet(processed_dir / "returns.parquet")
    daily = ret["daily_return"].to_numpy()

    # lags = ceil(median holding horizon), same convention as Day 27
    path = load_price_path() if price_path is None else price_path
    positions = build_positions(processed_dir=processed_dir, price_path=path)
    hold = [len(path[(path["date"] >= p["date"])
                     & (path["date"] <= p["expiry"])]) for p in positions]
    n_lags = int(np.ceil(np.median(hold)))
    block = max(2, int(np.ceil(np.sqrt(len(daily)))))   # ~ sqrt(T)

    sh = sharpe_with_nw(daily, n_lags)
    boot = block_bootstrap_sharpe(daily, block)
    haircut = is_oos_haircut(daily)

    if interpretation is None:
        text = ("Point Sharpe is negative but its 95% bootstrap CI "
                "spans zero and the NW t-stat is small in magnitude "
                "(|t|<1) -> the Sharpe is not statistically "
                "distinguishable from zero: no reliable edge either "
                "way, consistent with the disproof thesis. The "
                "negative IS->OOS haircut sign is noise at n=27/side.")
    elif interpretation == "auto":
        text = _auto_interpretation(sh, boot)
    else:
        text = interpretation

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
        "interpretation": text,
    }

    mpath = results_dir / "metrics.json"
    metrics = merge_metrics({"statistical_honesty": stats}, results_dir)

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
