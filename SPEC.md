# Options Vol-Arb — Implied-Vol Surface & Delta-Hedged Backtest

## Thesis

Build an arbitrage-free implied-volatility surface from index options chains, then run a
delta-hedged volatility-arbitrage backtest that trades the implied-vs-realized vol spread.

The headline deliverable is **not** "found alpha." On liquid index vol, the variance risk
premium is thin-to-zero after costs, margin, and discrete-hedging error. The deliverable is
a **rigorous, tail-honest disproof**: correctly demonstrate whether exploitable edge survives,
with Greeks-based PnL attribution that *reconciles* and return statistics that don't flatter a
negatively-skewed payoff. A clean, bulletproof negative result is a stronger signal than a
suspicious positive one — and the machinery (SVI no-arb calibration, full second-order Greeks
attribution, walk-forward backtest) is the point.

All numbers must be real and reproducible from raw data. No hardcoded or faked results.

---

## Stack

- Python 3.11+, numpy, pandas, scipy, matplotlib
- Clean module layout + pinned `requirements.txt` (or `pyproject.toml`)
- Single-command reproduction: `python main.py` regenerates every figure and metric from raw data
- Fixed seed, pinned versions, CI runs the test suite

---

## Data

**Pin ONE source per purpose — do not straddle both.**

- **Backtest (primary): SPX historical options with real bid/ask** (e.g. a public Kaggle SPX
  options-chain dataset). Real quotes are required so the cost model and liquidity filters bite
  on observed data, not assumptions.
- **Surface demo (secondary): NSE NIFTY/BANKNIFTY bhavcopy** — settlement prices only, no true
  bid/ask. Use for the surface-fitting showcase, and state explicitly that its cost model would
  be *assumed*, not observed. Do not use it for the costed backtest.
- Underlying spot + risk-free proxy (T-bill / OIS) from a free source.

### Data integrity layer

- `data/raw/` is immutable; ship a **SHA256 manifest** so the pipeline is provably reproducible
  offline.
- **Quote-quality filter**, with logged drop counts → `results/data_quality.json`:
  - drop crossed quotes (bid > ask), zero-bid, stale (no update), spread > X% of mid.
- **Forward construction via put-call parity** (imply F per expiry) rather than spot + r — this
  handles carry and dividends correctly.
- **Flag liquidity-conditioned selection bias:** the illiquidity filter tends to drop deep-wing
  strikes, which are exactly the cheap-vega tails. Note this as a known bias on the wings.

---

## Part 1 — Implied-Vol Surface

1. **Clean the chain:** apply quote-quality filter, compute mid prices, keep liquid tenors/strikes.
2. **Invert Black-Scholes** to implied vol per option (robust root-finder; handle forwards and
   cost-of-carry from the parity-implied forward).
3. **Calibrate raw SVI** per maturity slice (a, b, rho, m, sigma).
4. **Enforce no-arbitrage:**
   - **No-butterfly (convexity):** Durrleman `g(k) ≥ 0` within each slice.
   - **No-calendar:** total variance `w(k, T)` monotone non-decreasing in `T` across slices on a
     shared k-grid. This is a *joint* constraint — slices are not independent.
   - Constrained optimizer (SLSQP) with penalty; **reject and log** any violating slice →
     `results/arb_violations.json` (count, max severity).
5. **Calibration param smoothness:** track the SVI param time-series; jumpy params across dates
   flag overfit/unreliable slices (also a no-arb sanity proxy).
6. **Output:** fitted arbitrage-free surface; per-slice RMSE vs market (printed); vol-smile plots
   vs market; 3D surface.

---

## Part 2 — Vol-Arb Backtest

### Signal (no lookahead)

- **Realized vol estimator:** Yang-Zhang (OHLC-based, lower variance), **trailing window only**.
- **Forecast:** HAR-RV (daily / weekly / monthly RV terms) — standard, defensible, no peeking.
- **Signal:** `IV − forecast(RV)`, z-scored per tenor bucket, ranked cross-sectionally. Trade the
  rich/cheap deciles.

### Trade & hedge

- **Rule:** sell rich vol / buy cheap vol (straddles or single options).
- **Delta-hedge** each position with the underlying. The edge mechanism is:
  `PnL(delta-hedged option) ≈ ½ ∫ Γ·S²·(σ_IV² − σ_RV²) dt`.
- **Hedge frequency is a parameter** (daily / N-times-daily / band-based) with a sensitivity
  sweep — discrete-hedging error is the **dominant variance source**, not just a cost line
  (Leland-style: error scales with `ΔS²/√(hedge interval)`).
- **Note the two distinct RV objects:** the RV driving realized PnL is **dollar-gamma-weighted**
  realized vol along each option's own path/strike — *not* the plain Yang-Zhang used in the
  signal. Break-even vol is per-position. Attribution must be per-leg or the residual won't close.

### Greeks PnL attribution (must reconcile)

Decompose daily PnL with **second-order vol Greeks** — short index vol is short skew, so the
killer spot↓/vol↑ move lives in vanna and volga, not gamma:

```
PnL_actual ≈ δ·ΔS + ½·Γ·ΔS² + vega·Δσ
           + vanna·ΔS·Δσ + ½·volga·Δσ²
           + θ·Δt + rho·Δr + residual
```

- **Reconciliation test:** assert `mean(|residual| / |PnL|) < threshold`. Plot residual
  distribution. Without vanna/volga, the residual blows out on exactly the event days that
  matter — this is the proof the attribution is real, not decorative.

### Position sizing & risk

- Size by **vega and gamma** exposure under explicit **inventory limits** (per-name + portfolio).
- Drawdown kill-switch.

### Costs

- Per-leg half-spread + commission + hedge slippage on the underlying.
- **Cost sensitivity sweep** (× 0.5 / 1 / 2) → PnL-vs-cost curve.

---

## Reporting — return-distribution honesty

Sharpe is the **wrong headline** for a short-vol strategy (it structurally flatters
negatively-skewed payoffs; deflated Sharpe corrects multiple-testing, not skew). Report as
**co-headlines**, not extras:

- Skew, kurtosis, CVaR(5%), max drawdown, Calmar, Sortino — **alongside** Sharpe.
- Tail stats at **trade/weekly** horizon too (daily aggregation hides the tail).
- **Event PnL table** — strategy return during Feb 2018 (Volmageddon), Mar 2020, Aug 2024, and
  any in-sample vol spikes. This is what a desk PM looks at first.

### Capital base (define before quoting any Sharpe)

- Returns are computed on **SPAN / Reg-T margin**, stated up front. An unstated denominator makes
  Sharpe meaningless (premium vs vega-notional vs margin give wildly different numbers).
- Report the **margin procyclicality interaction:** SPAN margin expands in stress, exactly when
  returns turn negative → the tail gets hit twice. Surface this with the event table.

### Alpha isolation

- Don't just eyeball the benchmark — **regress** strategy returns on a short-straddle / VRP
  factor: report alpha, Newey-West t-stat (lags = ceil(holding horizon)), and R². Define the
  factor construction explicitly (delta-hedged-rolled vs raw). This quantifies how much is just
  short-vol beta vs residual edge.

### Statistical honesty

- Sharpe with Newey-West SE; bootstrap CI on Sharpe.
- **Deflated Sharpe (Bailey / López de Prado)** — but only meaningful if the trial count `N` is
  honest (see Pre-registration).
- **IS vs OOS Sharpe haircut** reported as a labeled result — the degradation number is itself a
  deliverable.

### Pre-registration (researcher degrees-of-freedom)

- Declare the **single primary config** before looking at results. SVI-vs-SABR, regime splits,
  cost/hedge sweeps are all DoF — every one is labeled **robustness**, not a menu to pick the
  best from. The DSR trial count `N` must include these honestly or it's undercounted.

---

## Validation

- **IV inversion round-trip:** price → IV → price, error < 1e-6 (test).
- **Synthetic recovery:** feed known-σ data → recover σ. Proves the engine before touching real
  data.
- **SVI fit quality:** per-slice RMSE vs market, printed.
- **Attribution reconciliation:** residual threshold test (above).
- **SVI no-butterfly check:** unit test on `g(k) ≥ 0`.

---

## Delivery sequencing — ship v1 complete before v2

A done clean-negative beats a sprawling TODO. Sequence, don't pile up.

### v1 — Primary spec (must ship end-to-end)

Single pre-registered config; SVI; daily hedge; signal (HAR-RV / Yang-Zhang); delta-hedge engine;
**full Greeks attribution incl. vanna/volga with reconciliation test**; return-distribution
honesty (skew/kurt/CVaR/Calmar/Sortino + event table); margin-based capital base; one
Sharpe-replacement headline. Reproducible via `python main.py`.

### v2 — Robustness appendix (labeled, post-hoc)

SABR as second calibration (SVI-vs-SABR RMSE table); regime split (VIX terciles — does edge
survive high-vol?); cost & hedge-frequency sweeps; DSR with full honest `N`; bootstrap CIs.

---

## Deliverables

- **Repo layout:** `data/` (`raw/` immutable + manifest), `src/surface/`, `src/backtest/`,
  `src/greeks/`, `notebooks/`, `results/` (figures + `metrics.json`), `tests/`.
- **`results/metrics.json`** is the single source of headline numbers — the README and report
  pull from it; no hand-typed Sharpe anywhere.
- **`results/report.html`** — static, auto-generated from `metrics.json`. The artifact a reviewer
  actually opens. (No Streamlit, no live deps.)
- **README** framed around the disproof thesis: what it does, how to run end-to-end, the data
  source + manifest, headline result, and 2–3 embedded plots.
- **`make all` / `python main.py`** regenerates everything from raw data.

### Unit tests

- BS pricing / Greeks (incl. vanna, volga).
- IV inversion round-trip.
- Synthetic σ recovery.
- SVI no-butterfly (`g(k) ≥ 0`) check.
- Attribution reconciliation residual threshold.

---

## Explicit cuts

- **No live yfinance snapshot** (unreliable, redundant with the pinned dataset).
- **No Streamlit** (toy; replaced by static `report.html`).

---

## Rating context

- **Method axis: 10/10** with vanna/volga closing attribution and the return-distribution +
  alpha-isolation + pre-registration machinery in place. Nothing a desk quant asks goes
  unanswered.
- **Alpha axis: ~3/10 — and that is the thesis.** "No exploitable VRP on liquid index after
  costs/margin/discrete-hedging, proven with reconciling Greeks attribution and tail-honest
  stats" is a stronger hire signal than a suspicious 2.0 Sharpe.
- The only gap between spec-10 and deliverable-10 is finishing **v1**.
