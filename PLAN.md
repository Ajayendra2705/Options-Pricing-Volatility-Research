# Implementation Plan — Day-by-Day (v1)

**Pace assumption:** each "Day" = one focused 3–4 hr session. ~5 sessions/week →
**v1 = ~30 sessions ≈ 6 weeks part-time.** v2 appendix is days 31–38.

**Rule:** every day ends with a committed, runnable artifact + a passing test where applicable.
No "half-finished, fix tomorrow." Front-loaded gates (Day 1) can kill the plan early — that's
the point.

---

## Week 1 — Data gate + BS engine foundation

### Day 1 — DATA GATE (load-bearing, do not skip)
- Pull candidate SPX options dataset (Kaggle / public). Inspect raw rows.
- **Gate check:** real two-sided quotes? `bid > 0`, `ask > 0`, `bid < ask`, not just settlement/last.
- Coverage scan: missing expiries, stale stretches, gap dates. Sample quote-quality filter → drop rate.
- **Decision:** drop rate < ~40% and real quotes present → proceed. Else switch source (shorter
  clean window OK) before writing any engine code.
- **Deliverable:** `notebooks/00_data_audit.ipynb` + `results/data_quality.json` (drop counts, date range).

### Day 2 — Repo scaffold + tooling
- Layout: `data/{raw,processed}/`, `src/{surface,backtest,greeks,utils}/`, `notebooks/`, `results/`, `tests/`.
- `requirements.txt` pinned, `README.md` stub, fixed seed util, `main.py` skeleton, CI config.
- `data/raw/` SHA256 manifest script.
- **Deliverable:** repo runs `python main.py` (no-op), `pytest` green (empty), committed.

### Day 3 — Black-Scholes pricing + first-order Greeks
- `src/greeks/black_scholes.py`: price (call/put), delta, gamma, vega, theta, rho. Forward-based.
- **Test:** known textbook values; put-call parity holds.
- **Deliverable:** `tests/test_bs_pricing.py` green.

### Day 4 — Second-order Greeks (the differentiator)
- Add **vanna** (∂Δ/∂σ), **volga/vomma** (∂vega/∂σ) — analytic.
- **Test:** finite-difference cross-check vs analytic, all Greeks, tight tol.
- **Deliverable:** `tests/test_greeks_fd.py` green. This is the line that separates the project.

### Day 5 — IV inversion + validation suite
- `src/greeks/iv_invert.py`: robust root-finder (Brent w/ Newton fallback), handle forwards/carry.
- **Tests:** round-trip price→IV→price err < 1e-6; synthetic known-σ recovery.
- **Deliverable:** `tests/test_iv_roundtrip.py`, `tests/test_synthetic_recovery.py` green.

---

## Week 2 — Clean data + SVI single slice

### Day 6 — Quote cleaning pipeline
- `src/surface/clean.py`: quote-quality filter (crossed/zero-bid/stale/wide-spread), mid prices.
- Logged drop counts → `results/data_quality.json` (real, not sample).
- **Deliverable:** cleaned chain parquet in `data/processed/`.

### Day 7 — Forward construction (put-call parity)
- Imply forward F per expiry from parity; back out carry/div. Compare to spot+r sanity.
- **Test:** parity-implied F stable across strikes per expiry.
- **Deliverable:** forward curve per date, plotted.

### Day 8 — IV surface from real data
- Invert whole cleaned chain → IV per option. Flag/handle deep-wing failures.
- **Flag** liquidity-conditioned selection bias on wings (documented).
- **Deliverable:** raw IV scatter per slice, plotted vs strike.

### Day 9 — Raw SVI calibration, one slice
- `src/surface/svi.py`: raw SVI (a,b,rho,m,sigma), least-squares fit one maturity.
- **Deliverable:** fitted smile vs market scatter, per-slice RMSE printed.

### Day 10 — SVI all slices + param time-series
- Loop all maturities. Track param time-series across dates (smoothness = overfit proxy).
- **Deliverable:** all-slice fit, RMSE table, param-stability plot.

---

## Week 3 — No-arbitrage (the hard part)

### Day 11 — No-butterfly constraint
- Durrleman `g(k) ≥ 0` per slice. Detect violations.
- **Test:** `tests/test_svi_butterfly.py` — known-bad params flagged.

### Day 12 — Constrained optimizer
- Refit SVI under butterfly constraint (SLSQP + penalty). Reject/log violators.
- **Deliverable:** arb-free per-slice fits, violation log.

### Day 13 — No-calendar constraint (joint)
- Total variance `w(k,T)` monotone non-decreasing in T on shared k-grid. Joint across slices.
- **Deliverable:** calendar-violation check, `results/arb_violations.json` (count, max severity).

### Day 14 — Surface assembly + QC
- Stitch arb-free slices → full surface. QC report.
- **Deliverable:** 3D surface plot, smile-vs-market panel, QC json.

### Day 15 — Surface buffer / catch-up
- Fix convergence issues, edge expiries, plotting polish. Lock Part 1.
- **Deliverable:** Part 1 reproducible via `main.py` stage 1.

---

## Week 4 — Signal + delta-hedge engine

### Day 16 — Realized vol estimator
- `src/backtest/realized_vol.py`: Yang-Zhang (OHLC), trailing window only. No lookahead.
- **Test:** recovers known σ on synthetic GBM path.

### Day 17 — HAR-RV forecast
- HAR (daily/weekly/monthly RV terms). Fit, forecast next-period RV.
- **Deliverable:** forecast vs realized plot, in-sample fit stats.

### Day 18 — Signal construction
- `IV − forecast(RV)`, z-score per tenor bucket, cross-sectional rank → rich/cheap deciles.
- **Pre-register** the single primary config here (write to `config/primary.yaml`).
- **Deliverable:** signal time-series, decile assignment.

### Day 19 — Delta-hedge engine core
- `src/backtest/engine.py`: open position, daily delta-hedge with underlying, mark-to-market.
- Hedge frequency = parameter (default daily).
- **Deliverable:** single-position hedged PnL path, sanity-plotted.

### Day 20 — Per-leg PnL + dollar-gamma-weighted RV
- Track per-leg PnL. Compute dollar-gamma-weighted realized vol per position (≠ signal RV).
- **Deliverable:** per-leg PnL ledger, break-even vol per position documented.

---

## Week 5 — Attribution + portfolio + costs

### Day 21 — Greeks PnL attribution
- Decompose: δ + ½ΓΔS² + vega·Δσ + vanna·ΔSΔσ + ½volga·Δσ² + θΔt + rho·Δr + residual.

### Day 22 — Reconciliation test (proof-of-life)
- Assert `mean(|residual|/|PnL|) < threshold`. Plot residual dist. Chase leaks until it closes.
- **Test:** `tests/test_attribution_reconcile.py` green on real data. **This is the gate that
  validates the whole project.**

### Day 23 — Portfolio assembly + sizing
- Size by vega + gamma under inventory limits (per-name + portfolio). DD kill-switch.
- **Deliverable:** full portfolio PnL series.

### Day 24 — Costs
- Per-leg half-spread + commission + hedge slippage. Apply to backtest.
- **Deliverable:** gross vs net PnL comparison.

### Day 25 — Capital base + returns
- Returns on SPAN/Reg-T margin (stated). Margin procyclicality flag in stress.
- **Deliverable:** margin-based return series, denominator documented.

---

## Week 6 — Reporting + polish + ship v1

### Day 26 — Return-distribution honesty
- Skew, kurtosis, CVaR(5%), max DD, Calmar, Sortino + Sharpe — co-headlines. Trade/weekly horizon too.
- **Deliverable:** stats block → `results/metrics.json`.

### Day 27 — Event PnL table + alpha isolation
- Event table: Feb 2018, Mar 2020, Aug 2024, in-sample spikes.
- Regress returns on short-straddle/VRP factor → alpha, Newey-West t, R².
- **Deliverable:** event table, regression output in metrics.json.

### Day 28 — Stats honesty
- Sharpe + Newey-West SE, bootstrap CI, IS-vs-OOS haircut (labeled result).
- **Deliverable:** full stats in metrics.json.

### Day 29 — report.html + README
- Static `report.html` auto-gen from `metrics.json` (no live deps).
- README framed on disproof thesis; embed 2–3 plots; pull numbers from metrics.json (no hand-typing).

### Day 30 — Full reproduce + lock v1
- `python main.py` raw→results end-to-end, clean clone, fixed seed. All tests green. CI passes.
- **Deliverable:** v1 shipped, reproducible, committed/tagged `v1`.

---

## v2 — Robustness appendix (Days 31–38, labeled post-hoc)

| Day | Work |
|-----|------|
| 31–32 | SABR second calibration; SVI-vs-SABR RMSE table |
| 33–34 | Regime split (VIX terciles) — does edge survive high-vol? |
| 35 | Cost sensitivity sweep (×0.5/1/2) → PnL-vs-cost curve |
| 36 | Hedge-frequency sweep (daily/N-daily/band) → variance sensitivity |
| 37 | Deflated Sharpe with honest N (incl. all robustness trials); bootstrap CIs |
| 38 | Final report/README update, tag `v2` |

---

## Critical path & risk

- **Day 1 data gate** — if it fails, everything downstream shifts. Resolve before Day 2.
- **Day 13 calendar no-arb** — joint optimizer convergence is fiddly; Day 15 buffer absorbs slip.
- **Day 22 reconciliation** — the make-or-break. If residual won't close, it's usually a sign
  convention or a missing path-dependent term, not a tuning problem. Budget Day 23 to bleed into
  this if needed.

**Honest total: v1 ≈ 6 weeks part-time. Don't promise less.** Buffers are at Days 15 and 23.
