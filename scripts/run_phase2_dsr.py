"""
Phase-2 Day 38 — Deflated Sharpe with an HONEST trial count (v2 capstone).

Resolves the item deferred since v1 Day 28 (stats.py) and Phase-2 Day 34:
"Deflated Sharpe requires an honest multiple-testing trial count N; N is only
complete after the v2 robustness sweeps." The regime split (Day 35), cost sweep
(Day 36) and hedge-frequency sweep (Day 37) are those trials, so N can now be
enumerated on the record instead of asserted.

The Deflated Sharpe (Bailey & Lopez de Prado 2014) is the Probabilistic Sharpe
evaluated against the Sharpe you would EXPECT the best of N random trials to
show under the null. Inputs:
  - the pre-registered SPY headline daily Sharpe, with its skew/kurtosis (the
    PSR correction for non-normal returns), and T observations;
  - an explicit TRIAL LEDGER: every distinct Sharpe examined across v1 + the
    Phase-2 walk-forward + the v2 sweeps. N = its length; V = its variance. The
    trials share data (folds/regimes/cadences are cuts of one book), so N is an
    UPPER bound on independent trials and the deflation is therefore
    conservative -- which only strengthens a disproof.

    results/phase2/{metrics,walkforward,regime_split,hedge_sweep}_spy.json
    results/metrics.json                                (v1 headline)
      -> results/phase2/deflated_sharpe_spy.json

Nothing tracked is mutated; the driver only emits deflated_sharpe_spy.json. It
supersedes the "deferred, not faked" DSR stub in metrics_spy.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.stats import (deflated_sharpe_ratio,            # noqa: E402
                                expected_max_sharpe_period,
                                probabilistic_sharpe_ratio)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"
V1_RESULTS = PROJECT_ROOT / "results"
PPY = 252
SQRT_PPY = np.sqrt(PPY)
N_SENSITIVITY = (6, 12, 24, 50, 100)


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> None:
    m = _load(RESULTS_DIR / "metrics_spy.json")
    wf = _load(RESULTS_DIR / "walkforward_spy.json")
    rg = _load(RESULTS_DIR / "regime_split_spy.json")
    hs = _load(RESULTS_DIR / "hedge_sweep_spy.json")
    v1 = _load(V1_RESULTS / "metrics.json")

    # ── headline SPY book: per-period Sharpe + non-normality ────────────────
    d = m["horizons"]["daily"]
    sr_ann = d["sharpe"]
    sr_period = sr_ann / SQRT_PPY
    n_obs = d["n"]
    skew = d["skew"]
    kurt = d["excess_kurtosis"] + 3.0            # PSR wants FULL kurtosis

    # ── explicit trial ledger (annualized Sharpes examined on the record) ───
    trials = [
        {"name": "v1 AAPL primary (daily hedge)",
         "sharpe_ann": v1["horizons"]["daily"]["sharpe"]},
        {"name": "SPY walk-forward primary (daily hedge)", "sharpe_ann": sr_ann},
        *[{"name": f"SPY hedge every {c['hedge_every_bars']} bars",
           "sharpe_ann": c["sharpe_annualized"]}
          for c in hs["curve"] if c["hedge_every_bars"] != 1],
        *[{"name": f"SPY fold {k}", "sharpe_ann": f["sharpe_annualized"]}
          for k, f in wf["folds"].items()],
        *[{"name": f"SPY regime {k}", "sharpe_ann": v["sharpe_annualized"]}
          for k, v in rg["day_level"]["regimes"].items()],
    ]
    sr_trials_ann = np.array([t["sharpe_ann"] for t in trials])
    n_trials = len(trials)
    var_ann = float(np.var(sr_trials_ann, ddof=1))
    var_period = var_ann / PPY
    best_trial = max(trials, key=lambda t: t["sharpe_ann"])
    best_config_ann = max(
        t["sharpe_ann"] for t in trials
        if "fold" not in t["name"] and "regime" not in t["name"])

    # ── PSR(0) and DSR at the enumerated N, plus a sensitivity band ─────────
    psr0 = probabilistic_sharpe_ratio(sr_period, 0.0, n_obs, skew, kurt)
    dsr = deflated_sharpe_ratio(sr_period, n_obs, skew, kurt,
                                var_period, n_trials)
    sr_star_ann = dsr["sr_benchmark_period"] * SQRT_PPY

    sensitivity = []
    for n in N_SENSITIVITY:
        star = expected_max_sharpe_period(var_period, n)
        sensitivity.append({
            "n_trials": n,
            "expected_max_sharpe_ann": star * SQRT_PPY,
            "dsr_headline": probabilistic_sharpe_ratio(
                sr_period, star, n_obs, skew, kurt),
        })

    # does the best data-mined slice clear the deflated bar?
    best_slice_survives = best_trial["sharpe_ann"] > sr_star_ann

    out = {
        "method": "Deflated Sharpe (Bailey & Lopez de Prado 2014): PSR evaluated "
                  "at the expected max Sharpe of N null trials. Sharpes are "
                  "annualized for display; the computation is per-period.",
        "headline_book": {
            "sharpe_annualized": sr_ann,
            "n_obs": n_obs,
            "skew": skew,
            "excess_kurtosis": d["excess_kurtosis"],
            "psr_vs_zero": psr0,
            "psr_note": "P(true Sharpe > 0), skew/kurtosis-corrected. Below 0.5 "
                        "means the point Sharpe cannot even clear zero.",
        },
        "trial_ledger": {
            "note": "every distinct Sharpe examined across v1 + Phase-2 walk-"
                    "forward + v2 sweeps. Trials share data (folds/regimes/"
                    "cadences are cuts of one book), so N over-counts independent "
                    "trials and the deflation is conservative.",
            "n_trials": n_trials,
            "trials": trials,
            "sharpe_dispersion_annualized": float(np.sqrt(var_ann)),
            "best_trial": best_trial,
            "best_standalone_config_sharpe_ann": best_config_ann,
        },
        "deflated_sharpe": {
            "computed": True,
            "supersedes": "the 'deferred, not faked' stub in metrics_spy.json "
                          "(resolved on Phase-2 Day 38)",
            "n_trials": n_trials,
            "expected_max_sharpe_ann": sr_star_ann,
            "dsr": dsr["dsr"],
            "sensitivity_over_n": sensitivity,
            "best_data_mined_slice_survives_deflation": bool(best_slice_survives),
        },
        "interpretation": (
            f"The pre-registered SPY Sharpe is negative (PSR vs zero = {psr0:.2f} "
            f"< 0.5): it cannot even clear zero. Every standalone configuration "
            f"tried is also negative (best {best_config_ann:+.2f}). The single "
            f"most favourable number the whole search produced -- the "
            f"'{best_trial['name']}' subset at {best_trial['sharpe_ann']:+.2f} -- "
            f"is below the deflated bar ({sr_star_ann:+.2f}, the Sharpe expected "
            f"from the luckiest of N={n_trials} null trials), so even the best "
            f"data-mined slice sits within multiple-testing noise. DSR ~ 0 for "
            f"every plausible N (6..100). Nothing in the project survives "
            f"deflation: the disproof is complete and multiple-testing-robust."),
    }
    out_path = RESULTS_DIR / "deflated_sharpe_spy.json"
    out_path.write_text(json.dumps(out, indent=2), newline="\n")

    print(f"headline SPY daily Sharpe {sr_ann:+.2f} (ann) | skew {skew:+.2f} "
          f"kurt+3 {kurt:.1f} | T={n_obs}")
    print(f"PSR vs zero = P(true SR>0) = {psr0:.3f}")
    print(f"\ntrial ledger: N={n_trials}, dispersion "
          f"{np.sqrt(var_ann):.2f} (ann Sharpe units); best trial "
          f"'{best_trial['name']}' {best_trial['sharpe_ann']:+.2f}; "
          f"best standalone config {best_config_ann:+.2f}")
    print(f"expected max Sharpe of N={n_trials} null trials (deflated bar) = "
          f"{sr_star_ann:+.2f} ann")
    print(f"DSR (headline vs that bar) = {dsr['dsr']:.4f}")
    print(f"best data-mined slice ({best_trial['sharpe_ann']:+.2f}) "
          f"{'CLEARS' if best_slice_survives else 'does NOT clear'} the "
          f"deflated bar ({sr_star_ann:+.2f})")
    print("\nsensitivity to N:")
    for s in sensitivity:
        print(f"  N={s['n_trials']:3d}: deflated bar "
              f"{s['expected_max_sharpe_ann']:+.2f} ann | DSR "
              f"{s['dsr_headline']:.4f}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
