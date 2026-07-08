"""
Day 26 — Return-distribution honesty.

Sharpe is the WRONG headline for a short-vol strategy: it structurally flatters
negatively-skewed payoffs (SPEC "Reporting").  So we report the tail/shape
statistics as CO-HEADLINES alongside Sharpe, at multiple horizons, on the
Day-25 margin-based return series (denominator = peak Reg-T margin, stated).

CO-HEADLINE BLOCK (per horizon): mean, std, annualized Sharpe, skew, excess
kurtosis, CVaR(5%) (expected shortfall = mean of the worst 5%), max drawdown,
Calmar (ann. return / |maxDD|), Sortino (downside-deviation Sharpe).

HORIZONS (daily aggregation hides the tail — SPEC):
  - daily   : per-bar margin returns (returns.parquet `daily_return`), ppy=252
  - weekly  : daily returns summed into calendar weeks, ppy=52
  - per_trade: each position's net PnL / capital base (10 trades) — non-
    annualized (overlapping, uneven holding); thin sample flagged.

rf = 0: the short window makes the cash rate negligible and financing is
already inside the engine PnL.  All returns are ARITHMETIC on a FIXED capital
base (not compounding), so horizon returns add.

OUTPUT: results/metrics.json (tracked, byte-stable).  This is the Day-26
deliverable; the event PnL table + alpha regression are Day 27.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TRADING_DAYS = 252
WEEKS = 52
CVAR_ALPHA = 0.05


# ── primitives ───────────────────────────────────────────────────────────────

def max_drawdown(cum: np.ndarray) -> float:
    """Peak-to-trough drawdown of a cumulative (level) series, in level units."""
    cum = np.asarray(cum, float)
    if cum.size == 0:
        return 0.0
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


def cvar(returns: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
    """Expected shortfall: mean of the worst ceil(alpha*n) returns.

    k-smallest (not quantile interpolation) so it is exact and well-defined
    for the small samples here.
    """
    r = np.sort(np.asarray(returns, float))
    n = r.size
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * n)))
    return float(r[:k].mean())


def sortino(returns: np.ndarray, ppy: float, target: float = 0.0) -> float:
    """Downside-deviation Sharpe.  Downside dev uses all N in the denominator
    (min(r-target,0)^2), the standard convention."""
    r = np.asarray(returns, float)
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd == 0:
        return float("nan")
    return float((r.mean() - target) / dd * np.sqrt(ppy))


def distribution_stats(returns: np.ndarray, ppy: float | None,
                       annualize: bool) -> dict:
    """Full co-headline block for one return array.

    ppy=periods-per-year; annualize=False -> Sharpe reported non-annualized
    (per_trade horizon), with ppy used only for Sortino scaling if given.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = r.size
    mean = float(r.mean()) if n else float("nan")
    std = float(r.std(ddof=1)) if n > 1 else float("nan")
    scale = np.sqrt(ppy) if (annualize and ppy) else 1.0
    sharpe = float(mean / std * scale) if (n > 1 and std > 0) else float("nan")
    cum = np.cumsum(r)                    # arithmetic, fixed base -> add up
    mdd = max_drawdown(cum)
    ann_ret = mean * ppy if (annualize and ppy) else mean * n
    calmar = float(ann_ret / mdd) if mdd > 0 else float("nan")
    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "sharpe_annualized": bool(annualize),
        "skew": float(sps.skew(r, bias=False)) if n > 2 else float("nan"),
        "excess_kurtosis": float(sps.kurtosis(r, fisher=True, bias=False))
        if n > 3 else float("nan"),
        "cvar_5pct": cvar(r),
        "max_drawdown": mdd,
        "calmar": calmar,
        "sortino": sortino(r, ppy if ppy else 1.0,
                           ) if (annualize and ppy) else
        sortino(r, 1.0),
        "worst": float(r.min()) if n else float("nan"),
        "best": float(r.max()) if n else float("nan"),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
    }


# ── runner ──────────────────────────────────────────────────────────────────

def run_metrics() -> dict:
    """Compute the co-headline distribution block at daily/weekly/per-trade
    horizons -> results/metrics.json."""
    ret = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    with open(RESULTS_DIR / "returns_summary.json") as fh:
        rsum = json.load(fh)
    with open(RESULTS_DIR / "costs_summary.json") as fh:
        csum = json.load(fh)

    capital_base = rsum["capital_base_usd"]

    # daily margin returns
    daily = ret["daily_return"].to_numpy()

    # weekly: sum daily returns into calendar weeks (arithmetic, fixed base)
    wk = (ret.set_index("date")["daily_return"]
          .resample("W").sum())
    weekly = wk.to_numpy()

    # per-trade: each position's net PnL / capital base
    per_trade = np.array([p["net_pnl"] / capital_base
                          for p in csum["positions"]], float)

    metrics = {
        "capital_base_usd": capital_base,
        "denominator": rsum["denominator"],
        "risk_free_rate": 0.0,
        "return_convention": "arithmetic on fixed capital base (non-compounding)",
        "net_pnl_usd": rsum["net_pnl_usd"],
        "net_return_on_capital": rsum["net_return_on_capital"],
        "horizons": {
            "daily": distribution_stats(daily, TRADING_DAYS, annualize=True),
            "weekly": distribution_stats(weekly, WEEKS, annualize=True),
            "per_trade": distribution_stats(per_trade, None, annualize=False),
        },
        "notes": {
            "headline": "Sharpe is NOT the headline for a short-vol book "
                        "(flatters negative skew); read skew/kurtosis/CVaR/"
                        "Calmar/Sortino as co-headlines.",
            "per_trade_sample": f"only {per_trade.size} trades — tail stats "
                                "are indicative, not significant (documented).",
            "sign": "net returns are negative after costs (Day 24), so Sharpe "
                    "and Sortino are negative by construction; shape stats "
                    "(skew/kurtosis) still describe the payoff.",
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    d = metrics["horizons"]["daily"]
    print(f"metrics: daily Sharpe {d['sharpe']:.2f} (ann), "
          f"skew {d['skew']:+.2f}, exc-kurt {d['excess_kurtosis']:+.2f}, "
          f"CVaR5% {d['cvar_5pct']:.4f}, maxDD {d['max_drawdown']:.4f}, "
          f"Calmar {d['calmar']:.2f}, Sortino {d['sortino']:.2f}")
    pt = metrics["horizons"]["per_trade"]
    print(f"  per-trade (n={pt['n']}): mean {pt['mean']:+.4f}, "
          f"skew {pt['skew']:+.2f}, win-rate {pt['win_rate']:.0%}, "
          f"worst {pt['worst']:.4f}")
    print(f"-> {RESULTS_DIR / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    run_metrics()
