# HANDOFF — Options Vol-Arb Project

> Living doc. Updated each session. Context for resuming work.

---

## Project Summary

Implied-vol surface (SVI, arb-free) + delta-hedged vol-arb backtest on SPY options.
Thesis: **disprove** exploitable VRP on liquid index after costs/margin/discrete-hedging.
Full spec: [SPEC.md](file:///c:/Users/ajiit/Desktop/Future/Projects/Options%20Trading/SPEC.md)
Full plan: [PLAN.md](file:///c:/Users/ajiit/Desktop/Future/Projects/Options%20Trading/PLAN.md)

---

## Current State

### Day 1 — DATA GATE (Done)

**Status:** Completed. Used AAPL shorter clean window as permitted by PLAN.md.

**Done:**
- [x] Repo directory structure created
  - `data/{raw,processed}/`, `src/{surface,backtest,greeks,utils}/`, `notebooks/`, `results/`, `tests/`, `config/`, `scripts/`
- [x] Data audit script: `notebooks/00_data_audit.py`
  - Auto-detects column names (handles multiple dataset formats)
  - Normalizes wide→long format
  - Runs 4 quality filters: missing, zero-bid, crossed, wide-spread
  - Outputs `results/data_quality.json`
  - Prints gate decision (PROCEED/MARGINAL/FAIL)
- [x] SHA256 manifest script: `scripts/sha256_manifest.py`
  - Generate + verify modes
  - Ensures `data/raw/` immutability

**Done:**
- [x] **Dataset download** — Pivot to real AAPL options data (June 2023)
  - Kaggle SPY data unavailable (auth limits); utilized `download_dolthub.py`
  - Shorter clean window satisfies `PLAN.md` fallback clause
  - Placed in `data/raw/aapl_options.parquet`
  - Ran: `python notebooks/00_data_audit.py` (Passed Gate: 15% drop rate)
  - Ran: `python scripts/sha256_manifest.py` (Generated Manifest)

**Gate criteria:**
- Real two-sided quotes (bid > 0, ask > 0, bid < ask) ✓ in at least some rows
- Drop rate < 40% after quality filters
- Date range covers key events (WAIVED: Pivot to shorter clean window (June 2023) invoked per `PLAN.md` fallback)

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ticker | AAPL (Pivot) | SPY Kaggle data unavailable. AAPL on DoltHub used to satisfy real-data gate logic (shorter clean window). |
| Audit format | `.py` script (not `.ipynb`) | Reproducible, no Jupyter dep |
| Communication | Ultra caveman mode | Token efficiency |

---

### Day 2 — Repo scaffold + tooling (Done)

**Status:** Completed.
- `requirements.txt` pinned
- `README.md` stub
- Fixed seed utility (`src/utils/seed.py`)
- `main.py` skeleton
- CI config (`.github/workflows/ci.yml`)
- `pytest` green (placeholder test)

### Day 3 — Black-Scholes pricing + first-order Greeks (Done)

**Status:** Completed.
- `src/greeks/black_scholes.py` — forward-based core (Black-76): `price`, `delta`, `gamma`, `vega`, `theta`, `rho` on (F, K, T, sigma, r, cp)
- Spot wrappers `*_spot` on (S, K, T, sigma, r, q, cp) via F = S·e^((r−q)T); standard textbook Greeks
- Vectorized (numpy); sigma·√T = 0 collapses to discounted intrinsic (no nan — needed for Day 5 IV inversion edges)
- `tests/test_bs_pricing.py` (13 tests): Hull textbook prices + Greeks, put-call parity grid (forward + spot w/ carry), spot↔forward consistency (chain rule), zero-vol/deep-ITM limits
- Full suite: 14 passed

### Day 4 — Second-order Greeks (Done)

**Status:** Completed.
- `vanna` (∂²V/∂F∂σ = −df·φ(d1)·d2/σ) + `volga` (vega·d1·d2/σ) added to `black_scholes.py`, forward core + spot wrappers. cp-independent. Zero-vol limit → 0, finite.
- `tests/test_greeks_fd.py` — FD cross-check ALL Greeks, tight tol (rtol 1e-6):
  - 1st-order (Δ, vega, θ, ρ) vs central FD of price
  - 2nd-order (Γ, vanna, volga) vs central FD of analytic 1st-order fns; vanna Schwarz symmetry (∂Δ/∂σ = ∂vega/∂F); Γ vs price 2nd-diff (looser)
  - Grid: {80,100,120} × T{0.05,0.5,2} × σ{0.1,0.3,0.8} × cp±, forward + spot(carry)
  - Sign-structure tests (vanna sign flip across forward-moneyness, volga ≥ 0, zero at d1·d2=0)
- Also FD-verified Day 3 Greeks before starting (8-decimal match)
- Full suite: 288 passed

### Day 5 — IV inversion + validation suite (Done)

**Status:** Completed.
- `src/greeks/iv_invert.py` — `implied_vol` (Black-76 forward core) + `implied_vol_spot` (carry via forward)
  - Newton (vega-based, σ-step convergence 1e-12) → Brent bracket fallback [1e-9, 5.0]
  - No-arb bounds: below discounted intrinsic / above σ→∞ limit → nan; at intrinsic → 0.0
  - Vectorized via `np.frompyfunc` over scalar core
- `tests/test_iv_roundtrip.py` — 160-pt grid roundtrip: reprice err < 1e-6 (PLAN gate) always; σ recovery rel 1e-6 where time value resolvable (deep-ITM: tv ~ ulp(p) → σ only ~1e-4, double-precision floor, documented in test); no-arb nan/zero-vol edges; spot+carry
- `tests/test_synthetic_recovery.py` — quadratic smile recovery, 500-pt seeded random sweep, tenor structure
- Real-data smoke: 46/46 June-2023 AAPL quotes inverted, median 0.28 vol pts vs vendor IV (S/r guessed; parity forwards come Day 7)
- Full suite: 458 passed

### Day 6 — Quote cleaning pipeline (Done)

**Status:** Completed.
- `src/surface/clean.py` — pure `clean_chain(df)` core + `run_cleaning()` CLI
  - Filters: missing / zero_bid / crossed / wide (>50% of mid) / **stale** (identical bid+ask vs same contract's previous date) / expired (T≤0)
  - Union drop semantics; per-filter counts pre-union
  - Normalizes any alias schema → canonical (date, expiry, strike, option_type C/P, bid, ask); adds `mid`, `T` (ACT/365)
  - Outputs `data/processed/chain_clean.parquet` + appends real counts under `"cleaning"` in `results/data_quality.json`
- Wired into `main.py --stage clean` (runs in `all` too)
- Real run: 714 → 607 clean, 15.0% drop (matches Day 1 audit exactly; 6 stale rows all overlap other filters)
- `tests/test_clean.py` — per-filter fixtures, stale same-contract logic, union no-double-count, T≤0, schema errors, real-parquet end-to-end invariants
- Full suite: 473 passed

### Day 7 — Forward construction (Done)

**Status:** Completed.
- `src/surface/forwards.py` — parity forwards, **ATM-window design**:
  - df fixed from external r (5.25%, June-2023 3M T-bill); F = median of per-strike F_k = K+(C−P)/df over 5 strikes w/ smallest |C−P|. Rate misspec ±200bp moves F < 2c (tested).
  - **Key finding:** all-strike regression implies df>1 (r ≈ −10%) on AAPL — American deep-ITM put EEP steepens the parity slope. ATM window sidesteps; `r_implied` kept as diagnostic only. Documented in module docstring.
  - Carry from forward term structure: q = r − d(lnF)/dT — no spot needed (vendor chain has no underlying column; replaces PLAN's spot+r sanity, documented waiver)
- Real run: 15 slices, F ∈ [178.22, 185.40] (tracks AAPL June-2023 path), worst ATM F std/F 0.29%; **q_implied +0.40%/+0.50% on data-rich dates ≈ AAPL actual div yield 0.5%**
- Deliverables: `data/processed/forwards.parquet`, `results/plots/forward_curves.png` (contango, r>q ✓)
- `tests/test_forwards.py` — exact synthetic recovery, rate-misspec insensitivity, noise, carry recovery (no spot), real-data stability (<0.5% ATM / <1.5% all-strike), levels, carry band
- Full suite: 484 passed

### Day 8 — IV surface from real data (Done)

**Status:** Completed.
- `src/surface/iv_surface.py` — inverts whole cleaned chain vs Day-7 forwards (one F/df per slice)
  - Per-row status: ok / below_intrinsic / above_upper / no_solution — failures **flagged, never dropped**
  - `log_moneyness` = ln(K/F) column; wing-coverage diagnostic (quoted range / ±4σ√T band)
  - **Liquidity-conditioned wing selection bias documented** in module docstring (zero-bid/wide-spread cleaning removes illiquid wings → surviving wing IVs conditioned on liquidity; quantified per slice)
- Real run: 607 quotes → **97.5% success** (592 ok, 15 below_intrinsic — all wing/EEP artifacts), IV ∈ [0.163, 0.962], ATM ~20-23%
- Scatter panels per date → `results/plots/iv_scatter_<date>.png` (gitignored, regenerable). Classic skew; C/P IVs agree near/below F, split above F = American ITM-put EEP in vol space (matches Day-7 finding)
- `tests/test_iv_surface.py` — full-path synthetic smile recovery (Days 5+6+7+8 integrated), failure classification, real-data: success >90%, IV band, C/P parity consistency near ATM (median <2 vol pts), failures live in wings
- Full suite: 491 passed

### Day 9 — Raw SVI calibration, one slice (Done)

**Status:** Completed.
- `src/surface/svi.py` — raw SVI (Gatheral): `svi_total_variance`, `svi_iv`, `fit_svi_slice`
  - LS in total-variance space, bounded trf, 6-point multistart over (m,σ) (loss multimodal there)
  - `otm_side` helper: fits **OTM quotes only** (puts K<F, calls K≥F) — EEP-clean side per Day-7/8 finding
  - Butterfly/calendar no-arb NOT enforced (Days 11–13 by design)
- Real fit (most-quoted slice 2023-06-02 → 06-16, 24 OTM quotes): **RMSE 0.40 vol pts**; params a=−0.0089 b=0.065 ρ=−0.15 m=−0.024 σ=0.169; global min w = +0.0019 > 0 ✓; plot `results/plots/svi_fit_*.png`
- `tests/test_svi.py` — formula anchors (w at k=m, asymptotic wing slopes), exact curve recovery (RMSE <1e-6), noise robustness, bounds, real-slice RMSE <2 vol pts + w≥0
- Full suite: 497 passed

### Day 10 — SVI all slices + param time-series (Done)

**Status:** Completed.
- `fit_all_slices` + `plot_param_stability` + `run_svi_all` added to `svi.py` (CLI: `python -m src.surface.svi --all`)
- Real run: **14/15 slices fitted** (skip: 06-09→07-28, 4 OTM pts < MIN_POINTS=6)
  - **median RMSE 0.33 volpts, max 0.51**, zero negative-w slices
  - Params smooth across dates for T ≳ 0.07; shortest tenor (06-23 exp, T=0.03, 11 pts) shows param trade-off (ρ→−0.47, σ→0.11) while ATM IV stays stable — raw-SVI degeneracy documented, motivates Day 11–13 constraints
- Outputs: `data/processed/svi_params.parquet`, `results/plots/svi_param_stability.png`
- Tests: synthetic 2-slice fit_all, thin-slice skip, real all-slice gates (median <1 / max <3 volpts, bounds, w≥0), per-expiry ATM-IV stability across dates
- Full suite: 501 passed

### Day 11 — No-butterfly constraint (Done)

**Status:** Completed.
- `src/surface/no_arb.py` — `durrleman_g` (analytic w′ = b(ρ+d/R), w″ = bσ²/R³), `check_butterfly` grid scan (k∈±1.5, 1001 pts), `check_all_slices`, CLI runner
- g = −inf where w ≤ 0 (negative variance flagged, not masked)
- Real run: **14/14 slices arb-free**, min g ≥ 0.023 — raw fits clean on this window; Day 12 constraint still needed as guarantee
- `tests/test_svi_butterfly.py` — **Axel Vogt known-bad params flagged** (PLAN gate); flat-smile g≡1; analytic derivs vs FD; **independent Breeden–Litzenberger detector** (FD density from Black calls) agrees with g's verdict + >99% pointwise sign match
- Full suite: 509 passed

### Day 12 — Constrained optimizer (Done)

**Status:** Completed.
- `fit_svi_constrained` + `refit_all_constrained` + `run_constrained_refit` in `no_arb.py` (CLI `--refit`)
  - Unconstrained first; if arb-free → constraint inactive (zero cost). Else SLSQP w/ vector ineq g(k)≥margin, w(k)≥floor on 201-pt grid; escalating hinge-penalty L-BFGS-B fallback
  - **G_MARGIN=2e-4**: SLSQP satisfies constraints only to ~ftol — landed −2.8e-5 below zero vs the stricter 1001-pt post-check until margin added
  - Gotcha fixed: mixed-dtype DataFrame cols made `~bool_col` arithmetic (−2 each) — astype(bool) before negation
- Vogt arbitrable market: unconstrained fit inherits violation; constrained → arb-free at RMSE < 0.6 volpts (G-J report ~0.3 for this case)
- Real run: 14/15 fitted, 0 pre / 0 post violations (constraint unbinding on this window), median RMSE 0.33 volpts unchanged
- Outputs: `svi_params_constrained.parquet` + `results/svi_butterfly_log.json` (violation log)
- Full suite: 515 passed

### Day 13 — No-calendar constraint (Done)

**Status:** Completed.
- `fit_svi_constrained` generalized with `w_floor` (previous expiry's w on constraint grid, ≥ floor + CAL_MARGIN=1e-5)
- `check_calendar` (pairwise consecutive expiries per date, severity = max w decrease), `fit_all_joint` (per date, short→long T, sequential floors), `run_arb_check` CLI `--joint`
- Coordinate: per-expiry forward moneyness k=ln(K/F_T) (Gatheral convention), documented
- **Real violation found & fixed:** 06-12 pair 07-07→07-28 violated between constraint-grid nodes (201-pt fit grid vs 1001-pt check grid) — joint fit now constrains on full check grid. Second instance of the "optimizer grid coarser than check grid" bug class (Day 12 = margin, Day 13 = density).
- Real run: 14 slices, **0 butterfly / 0 of 9 calendar pairs violated, max severity 0**, median RMSE 0.33 volpts unchanged
- Outputs: `svi_params_joint.parquet`, **`results/arb_violations.json`** (PLAN deliverable, tracked)
- `tests/test_svi_calendar.py` — synthetic violating pair detected w/ exact severity, clean pair passes, floor-refit clears floor + stays butterfly-free, joint fit on violating synthetic surface → clean, real-surface gates
- Full suite: 522 passed

### Day 14 — Surface assembly + QC (Done)

**Status:** Completed.
- `src/surface/assemble.py`: `VolSurface` per quote date — Day-13 joint SVI slices + Day-7 forwards, queryable `w(k,T)` / `iv(k,T)` / `iv_strike(K,T)`
- T-interpolation: **linear in total variance at fixed forward moneyness k** (calendar-monotone by construction); flat-IV extrapolation outside node range (`w·T/T_edge`, monotone both directions)
- Forward curve: ln F linear in T between implied-forward nodes (constant carry), edge slope extrapolated
- Interpolated-slice arbitrage NOT assumed from theory — QC re-checks Durrleman g via FD on 21 T-slices per date + calendar monotonicity on dense grid
- Real run: 5 dates / 14 slices, **interp butterfly clean (worst g +1.2e-2), interp calendar clean**, vs market: median RMSE 0.39 volpts, 97.4% of OTM quotes within 1 volpt, max abs err 2.5 volpts (short-T wing)
- Outputs: **`results/surface_qc.json`** (PLAN deliverable, tracked), 5× `surface_3d_<date>.png` + 5× `smile_vs_market_<date>.png` (plots regenerable, gitignored)
- Smile panels show grey ITM-side quotes diverging above F — the documented American-EEP contamination, visibly excluded from fits
- `tests/test_assemble.py` — node exactness, linear-w interp, flat-IV extrap, calendar monotone everywhere, interpolated butterfly FD check, ln-F-linear forward + iv_strike consistency, real-surface QC gates
- Full suite: 533 passed

### Day 15 — Surface buffer / lock Part 1 (Done)

**Status:** Completed. No convergence issues outstanding (Day-13 buffer unneeded); buffer spent on one real fix + pipeline lock.
- **Fix (found by Day-14 independent verify):** forward curve dropped the valid 06-09/07-28 implied forward because its vol slice failed to fit (4 OTM pts < MIN_POINTS). `VolSurface` forward nodes now decoupled from vol nodes (`F_Ts` field, `build_surfaces` passes ALL date forwards) — was a 1.2e-3 relative F error beyond T=0.077 on that date
- `main.py --stage surface` wired: forwards → IVs → SVI diagnostics → joint arb-free fit → assembly + QC
- **Part 1 locked:** full `python main.py --stage all` rerun from raw data reproduces all 6 processed parquets + arb_violations.json + surface_qc.json **bit-identically** (SHA256 snapshot compare)
- `tests/test_pipeline.py` — stage wiring + cross-artifact consistency (forwards/iv/joint cover exactly the cleaned-chain slices; QC json ↔ parquet counts; 0 violations)
- README: stage usage + reproducibility note
- Full suite: 535 passed

### Hardening pass — full-audit caveat fixes (Done, post-Day-15)

**Status:** Completed. Full independent audit (Days 1–15) passed; every known caveat then fixed:
- **Thin-slice rescue:** `fit_points` in svi.py — OTM side; if < MIN_POINTS, augment with near-ATM ITM quotes (|k| ≤ ATM_AUGMENT_BAND=0.10, where EEP contamination < 0.3 volpts on real data). 06-09/07-28 now fits → **15/15 slices**, calendar pairs 9→10. Its unconstrained fit was **genuinely arbitrable** (BL min density −146, spiky σ=0.007/b=10.3) — detection flagged it, constrained refit repaired it ("pre-violations 1, post 0"): the Days 11–13 machinery proven on live data.
- **Interp made arb-free BY CONSTRUCTION:** flat-IV long-end extrapolation of the new slice broke Durrleman (g=−0.021 at 1.25·T_last — QC caught it). VolSurface now: interior = linear in normalized OTM option price at fixed k (convex combo of convex ordered price curves ⇒ static-arb-free), long end = flat total variance (same slice ⇒ safe), short end = flat-IV scaling down (QC-checked numerically). Inversion via vectorized bracketed Newton (assembly 4min→17s). QC: worst interp g **+2.0e-4**, all clean.
- **Object-dtype bug class killed at source:** `fit_ok` cast `astype(bool)` in all three fit-table producers (fit_all_slices, refit_all_constrained, fit_all_joint).
- **Pipeline completeness:** `run_constrained_refit` wired into `main.py --stage surface` (Day-12 deliverables regenerate).
- **Timestamp churn:** removed from data_quality.json cleaning section — all tracked results now byte-stable across identical reruns (verified: 10/10 outputs bit-identical on full rerun).
- **Verify suite moved into repo:** `scripts/verify/` (Days 3–14 independent checks + `audit_full.py` + `run_all.py` runner), paths portable. **10/10 pass.** verify_day11 upgraded: curvature-scaled FD tolerance + detection-consistency check (negative density ⟺ durrleman flag) + densities validated on the authoritative joint params.
- Full suite: 537 passed. vs market: max abs err 2.50→**1.44 volpts**, within-1-volpt 97.4→**98.0%**.

### Day 16 — Realized vol estimator

**Status:** Code completed, gated on synthetic tests. `src/backtest/realized_vol.py`: Yang-Zhang (overnight + k·open-close + (1−k)·Rogers-Satchell, k=0.34/(1.34+(n+1)/(n−1))), trailing-only (estimate at t uses bars t−n+1..t), close-to-close baseline, `realized_vol_table` (yz/cc × windows 10/21/63). Tests (8): recovers σ=0.30 on fine-step GBM, YZ variance < 0.75·CC (efficiency), drift-robust, inline-formula recompute 1e-12, **no-lookahead invariance** (future bars ×7 → estimates through t unchanged), NaN warm-up structure, validation errors. Wired as `main.py --stage backtest`.

**Data follow-up DONE (2026-07-03):** `data/raw/aapl_ohlc.parquet` — 375 rows, 2022-01-03..2023-06-30, SHA256 in manifest. Real RV: June-2023 YZ21 ≈ 14–17% (calm regime, as expected). Real HAR: in-sample R² 0.357, OOS expanding corr 0.463, betas d/w/m 0.10/0.18/0.34 (monthly dominant — normal for a noisy daily proxy). **Real signal: mean −4.75 volpts, ALL 14/15 slices negative** — expanding HAR trained through 2022's high-vol regime forecasts RV above June-2023 IV; signal says vol is "cheap" everywhere, more so at longer tenors. That's the regime-lag property of the pre-registered spec, recorded before PnL exists; interpretation belongs to Days 23+. Sides: 5 short_vol / 5 flat / 5 long_vol. Backtest-stage artifacts byte-stable across reruns. Suite: **571 passed, 0 skipped** (real-data tests active; test_clean now pins aapl_options.parquet instead of glob[0]).

### Day 17 — HAR-RV forecast

**Status:** Code completed, gated on synthetic tests (same pattern as Day 16). `src/backtest/har.py`:
- Daily variance proxy `v_t = 252·(overnight² + Rogers-Satchell_t)` — per-bar, drift-independent. Known ~10% downward discrete-monitoring bias on RS (documented in tests; cancels in the IV−RV signal z-score).
- Log-vol HAR: `ln σ_fwd ~ 1 + ln σ_d + ln σ_w(5) + ln σ_m(22)`, target = √(mean v over next h=21 bars). Half-variance correction `exp(ŷ+s²/2)` so forecasts target the mean, not the median.
- **Two forecast columns:** `har_insample` (diagnostics only) and `har_oos` (expanding window, at t fit only on rows whose target is realized by t, i.e. i ≤ t−h) — **Day-18 signal must consume `har_oos`**.
- Outputs: `data/processed/har_forecast.parquet`, `results/har_stats.json`, forecast-vs-realized plot. Wired into `--stage backtest` after realized vol.
- Tests (8): proxy recovers σ² (bias-aware one-sided band), constant-vol level unbiased vs own proxy scale, R²>0.45 on persistent log-vol-AR(1) synthetic (observed 0.504; OOS corr 0.627 with realized), OLS matches normal-equations recompute, no-lookahead invariance on expanding forecast, warm-up/target NaN structure. Full suite 553 passed, 2 skipped (real-data).

### Day 18 — Signal construction + pre-registration

**Status:** Code completed, gated on synthetic tests. **`config/primary.yaml` written and locked BEFORE any PnL exists** — genuine pre-registration. Key declared choices:
- `signal_raw = atm_iv − har_oos(h)` in volpts, ATM IV at k=0 from joint arb-free SVI, HAR horizon matched to tenor `h = clamp(round(252T), 5, 63)`.
- **Normalization: none in primary.** 5 quote dates → trailing per-bucket z has ≤ 4 obs = noise; z per tenor bucket (expanding, trailing-only, min 3 prior obs) computed as DIAGNOSTIC column only. Declared up front, not post-hoc.
- Ranking within quote date by raw signal desc: rank 1 → short_vol, last → long_vol, middle flat. PLAN's "deciles" collapse to ranks at a 3-slice cross-section — documented adaptation.
- Costs/margin/instrument also locked: ATM straddle nearest strike, daily delta hedge, half-spread fills + $0.65/contract, reg-T 20% proxy margin, margin-based capital.

`src/backtest/signal.py`: `build_signal(params, ohlc)` → per (date, expiry): atm_iv, bucket, h, rv_fcst, signal_raw, signal_z, rank, side. Unfitted slices dropped; NaN forecasts excluded from ranking (side=flat). `expanding_forecast` rewritten with incremental normal equations (O(n), 44s→8.5s test module; identical math — RSS via y'y − β'X'y). Tests (7): exact ATM-IV recovery on b=0 SVI, bucket/horizon clamping, manual signal recompute, rank/side per date, z trailing-only + truncation invariance (no lookahead), fit_ok exclusion, NaN-forecast handling. Suite: **560 passed, 2 skipped** (real-data).

**Pending:** real signal table needs `aapl_ohlc.parquet` (download in progress, resumable).

### Day 19 — Delta-hedge engine core

**Status:** Completed. `src/backtest/engine.py`: `Leg` (K, expiry, cp, qty, mark_vol, mult) + `run_hedged(dates, S, legs, r, q, hedge_every)` → daily self-financing ledger (`equity = cash + shares·S + V_opt`, starts at 0 = cumulative PnL). Marking at constant per-leg entry IV (keeps Day-20 attribution clean); hedge to flat delta every `hedge_every` bars (always at entry); expiry settles intrinsic to cash, dead book liquidates hedge; cash accrues at r ACT/365. **No costs yet by design** — ledger records `traded` shares so the Day-21 cost model bolts on without touching accounting. Rebalancing trades cash↔stock at the same price, so it can never move equity: self-financing by construction.

Tests (9): **exact identities** — synthetic forward (long C + short P, r=0) hedges to equity ≡ 0 at 1e-9 on every bar; flat path keeps exactly the entry premium; ledger algebra (equity column identity, Δcash = −traded·S, fully-settled end state). **Hedging physics** (GBM MC, SE-scaled asserts) — RV=IV → mean PnL ≈ 0; RV<IV → short vol wins; RV>IV → loses; hedge_every=5 inflates PnL dispersion ~√5 (Leland). Plus no-lookahead invariance + validation. Demo deliverable: `results/plots/engine_demo.png` (short 180 straddle, IV 25/RV 20, +$618 grind — classic short-gamma profile). Suite: **569 passed, 2 skipped**.

### Day 20 — Per-leg PnL + dollar-gamma-weighted RV

**Status:** Completed. Engine now emits a per-leg ledger (`run_hedged(..., return_legs=True)`): per (leg, date) value, delta, **dollar_gamma = qty·mult·Γ·S²**, vega, theta, and daily mark PnL (settlement included). `src/backtest/pnl.py`:
- `gamma_weighted_rv`: σ²_gw = Σ|$Γ_{t−1}|·ret²_t / Σ|$Γ_{t−1}|·dt_t (calendar ACT/365, same clock theta accrues on). **This is the RV that drives hedged PnL — not the signal's Yang-Zhang/HAR** (SPEC's two-RV distinction).
- `theta_gamma_pnl`: first-order per-bar hedged PnL ½$Γ_{t−1}(ret² − σ²_mark·dt) — seed of Day-23 attribution.
- `breakeven_report`: σ_gw **is** the position's break-even vol; short-gamma book profits iff σ_gw < σ_mark. Report: book mark, σ_gw, gap in volpts, per-leg PnL, sign-consistency flag.

Tests (7): per-leg marks sum exactly to book; **per-leg PnL + hedge-holding PnL reproduces equity bar-by-bar at 1e-9**; σ_gw recovers σ on GBM; theta-gamma twin tracks equity (corr>0.95, slope≈1, cumulative gap <15% premium); break-even sign-consistency >85% across 60 paths at RV=IV (first-order, not 100% — honest); flat path σ_gw=0 + each leg keeps its premium. Deliverable plot `results/plots/pnl_decomposition.png` (twin tracks equity, diverges only near expiry where higher-order terms bite). Demo breakeven: σ_gw 16.1% vs mark 25% → +$618, per-leg +139 (call) / +728 (put), hedge −249.

**Perf fix (durable):** `black_scholes.py` switched `scipy.stats.norm` → `scipy.special.ndtr` + local pdf (identical values, no per-call framework overhead): engine run 0.95s → 0.035s (27×). Verify suite re-run after the swap: **10/10**. Suite: **578 passed in ~72s**.

### Day 21 — Greeks PnL attribution

**Status:** Completed. (Correction: earlier stub here guessed "costs" — PLAN Day 21 is **attribution**; costs are Day 24.) `src/backtest/attribution.py`:
- `leg_attribution(dates, S, legs, sigma_by_leg, r, q)`: per leg, per bar, Taylor terms around the **previous close** (Greeks at S_{t−1}, τ_{t−1}, σ_{t−1}): δ·ΔS + ½Γ·ΔS² + θ·Δt + vega·Δσ + vanna·ΔS·Δσ + ½volga·Δσ² + ρ·Δr. Takes per-leg σ as float **or series** — engine marks are constant-vol today (vol terms exactly 0), but surface re-marking and Day-22 real-data reconciliation reuse this unchanged. ρ term identically 0 until a rate series exists (r constant in engine) — column kept because PLAN pre-registers the decomposition.
- `book_attribution(ledger, legs, ...)`: adds the two **exact** (no-Taylor-error) terms from the ledger — hedge holding PnL `shares_{t−1}·ΔS` and financing `cash_{t−1}·(e^{rΔt}−1)` — so `residual = actual − explained` is **purely the sum of per-leg option Taylor errors** (proved by test). `attribution_summary` → the numbers Day-22 will threshold.

Tests (9): delta term cancels hedge PnL **exactly** (1e-9) under daily hedging (same pricer both sides); book residual == Σ per-leg Taylor errors (1e-9); financing closes the ledger at r=5%; constant marks → vol columns identically zero; GBM straddle residual <10% premium cumulative (<5% excluding final pre-expiry bars where Γ blows up — known Taylor breakdown, documented); **time-varying σ on a manually re-marked leg: vega/vanna/volga switch on and residual <5% of Σ|PnL|**; no-lookahead invariance; validation. Demo `results/plots/attribution_demo.png`: θ +1124 / Γ −465 / δ+hedge exactly cancel / residual −41 on +618 total, concentrated in last bars before expiry. Suite: **587 passed**.

### Day 22 — Reconciliation gate on real data (PLAN: "validates the whole project")

**Status:** Completed — **gate GREEN on first real-data pass** (no leaks to chase; the Day-21 identity test already guaranteed the residual is pure per-leg Taylor error, so the only question was its size). `src/backtest/reconcile.py`:
- `build_positions()`: pre-registered book from `signal.parquet` — one ATM straddle per non-flat row (10 positions: 5 short_vol + 5 long_vol), strike = nearest listed strike with both sides quoted to the slice forward, marked at the **joint arb-free SVI vol at that strike**, qty ±1.
- **Rates per slice from real data:** r = market-implied discount from Day-7 forwards; carry q backed out of the traded forward (q = r − ln(F/S₀)/T) so the engine forward matches the market forward at entry — dividends handled implicitly, no external feed.
- **Hedge-path data extension:** `data/raw/aapl_ohlc_ext.parquet` (2023-07-03..08-31, 43 bars, DoltHub same source, manifest updated) — long-leg expiries (07-21/07-28/08-18) reach settlement. Deliberately a **separate raw file**: signal/HAR/RV inputs still end 2023-06-30, so the no-lookahead boundary between "data the signal saw" and "path the trade lives on" is physical. Original raw hashes untouched. (Additive change, documented per primary.yaml rules.)
- `run_reconcile()` → `results/attribution_reconcile.json` (tracked, byte-stable) + `results/plots/attribution_residuals.png`; wired into `main.py` backtest stage.

**Real numbers:** book residual abs share **14.4%** of Σ|daily PnL| (gate < 20%); worst position cumulative residual **7.3% of premium** (gate < 10%), 9/10 positions ≤ 2.8%. The worst is the 06-14→08-18 long straddle: +$84.7 residual on 2023-08-04's −4.80% earnings-gap bar — a one-day Taylor expansion undercounts a long-gamma gain on a big move (third-order, correct sign, correct location; physics, not leak). Total book PnL pre-costs: **+$3.87** (shorts +$505.8, longs −$501.9 — the two sides nearly cancel; costs arrive Day 24).

Tests (5, skip wholesale if real files missing): book matches pre-registration (10 positions, 5/5 sides, |ln(K/F)|<0.03, sane implied carry); engine ledger identity on every real path; **Day-21 residual identity re-proven on real data at 1e-9**; the gate itself; vol terms structurally zero under constant marks. Suite: **592 passed** (~27s clean; a 547s run was machine contention again).

### Day 23 — Portfolio assembly + sizing (Done)

**Status:** Completed. `src/backtest/portfolio.py`:
- **Sizing:** vega + gamma per-position limits (binding constraint wins). Defaults: `vega_limit=500` / `gamma_limit=5000`. All 10 positions gamma-bound (short-dated ATM gamma dominates vega scaling).
- **Portfolio limits:** gross vega + gross gamma caps with pro-rata clip. Defaults: `portfolio_gross_vega=3000` / `portfolio_gross_gamma=30000`. Portfolio gamma hit exactly 30000 (gamma binding across all positions).
- **Drawdown kill-switch:** 15% threshold on peak-to-trough. **Fired on 2023-06-15** — the first day after the last entry date, early losses from long-vol legs pushed portfolio equity past the kill threshold. Total PnL frozen at +$0.39 (raw without kill: −$3.04).
- **Kill-switch firing is correct and expected:** the unit-qty book (Day 22) was PnL-neutral (+$3.87 total); after risk-scaling to gamma budgets, positions are tiny (qty ~0.007–0.017) and the per-bar equity is ~$0.10 scale — the kill threshold on an equity curve that starts at 0 and never climbs above ~$1 is inherently tight. The kill is a feature: it documents that the VRP signal produces no robust edge after sizing, exactly the SPEC's disproof thesis.
- **Config additive change:** `config/primary.yaml` gains a `sizing:` block (risk limits only, signal unchanged). Documented per pre-registration rules.
- Outputs: `data/processed/portfolio.parquet` (54 bars, 2023-06-02..2023-08-18), `results/portfolio_summary.json`, `results/plots/portfolio_equity.png`
- Wired into `main.py --stage backtest` after `run_reconcile()`.
- `tests/test_portfolio.py` (10 tests): vega/gamma sizing to limits, binding constraint, portfolio pro-rata clip, DD kill-switch activation + non-activation, aggregation identity, sign consistency, real-data gate (10 positions, date range, max DD finite). Suite: **602 passed**.

### Day 24 — Costs (Done)

**Status:** Completed. `src/backtest/costs.py` applies the **pre-registered** cost model (config `costs:` block, locked Day 18) to the **unit-qty pre-registered book** (same 10 positions Day 22 gated) → gross vs net PnL:
- **Option entry half-spread:** cross ½·(ask−bid) per share on BOTH legs at entry (from entry-date `chain_clean` quotes), ×mult×|qty|. Always a positive drag (crossing costs both sides).
- **Commission:** $0.65/contract × 2 legs on the opening trade. Held to expiry → cash-settled intrinsic, no closing trade/commission (documented).
- **Hedge slippage:** config declares AAPL penny-wide, **zero underlying spread in primary**. Mechanism implemented + parametrized (`underlying_slippage_bps`, default 0) — robustness-only knob, does not touch primary.
- Config parsed via same block-slice trick as Day-23 sizing (`_load_cost_params`, primary.yaml's prose colons break strict YAML).

**Real numbers:** gross **+$3.87** − costs **$419.00** = **net −$415.13**. Half-spread $406 (two long-dated 07-28 thin slices dominate: $102.50 + $212.50 — realistic wide long-dated AAPL quotes), commission $13, hedge slippage $0. **This is the disproof punchline: the near-breakeven gross VRP edge is annihilated by realistic entry costs, net clearly negative.** Cost = 28% of total premium.

- Outputs: `results/costs_summary.json` (tracked, byte-stable — verified), `results/plots/gross_vs_net.png`. Wired into `main.py --stage backtest` after `run_portfolio()`.
- `tests/test_costs.py` (8 tests): half-spread formula + qty scaling, commission, non-negative drag both sides, hedge slippage zero@primary / scaled@bps>0, config parse, real-data gate (10 positions, net = gross − cost identity per-position + book, net < 0). Suite: **610 passed**.

### Day 25 — Capital base + returns (Done)

**Status:** Completed. `src/backtest/returns.py` fixes the return denominator (SPEC: "define capital base before quoting any Sharpe") on the **unit-qty pre-registered book**:
- **Reg-T margin model** (config `margin:` block): naked short-option req/share = `max(0.20·S − OTM, 0.10·S) + premium`; short straddle = larger naked leg + other leg's premium; long straddle = premium debit (fully paid). Recomputed **every bar at Sₜ, τₜ** → procyclical path.
- **Capital base = peak book Reg-T margin over the window = $27,106** (2023-06-15). Documented denominator string in the JSON.
- **Returns:** numerator = **net** book equity (engine gross − Day-24 entry costs at each entry bar); `net_return = net_pnl / capital_base`. Gross reported alongside. **Net −$415.13 → −1.53% on capital** (gross +0.014%).
- **Margin procyclicality — decontaminated:** the raw peak/entry (5.08×) is a *staggered-entry* artifact (2 straddles live 06-02 → 10 by 06-15), NOT stress, and is labeled as such. Genuine market-driven effect isolated two ways: (a) **per-position stress ratio** (single straddle's peak margin / entry margin) = max **1.63×** / mean 1.20× — pure spot move on a fixed leg; (b) **corr(ΔMargin, ΔEquity) = −0.09** computed ONLY on bars where the live set is unchanged (no entry/settlement jump) → weakly procyclical, correct sign (margin rises as equity falls — the "tail hit twice" the SPEC wants surfaced; Day-26 event table quantifies).
- Outputs: `data/processed/returns.parquet` (54 bars, per-bar margin/equity/return), `results/returns_summary.json` (tracked, byte-stable — verified), `results/plots/margin_returns.png`. Wired into `main.py --stage backtest` after `run_costs()`.
- `tests/test_returns.py` (8): naked-margin formula + floor-binds-deep-OTM, short-straddle rule, long-straddle debit, settled-leg zero, spot-stress raises short-straddle margin (procyclicality direction), qty scaling, real-data gate (capital base = peak ≥ entry, net_return = net_pnl/base identity, net < 0). Suite: **618 passed**.

### Day 26 — Return-distribution honesty (Done)

**Status:** Completed. `src/backtest/metrics.py` — co-headline distribution stats on the Day-25 margin-based return series (denominator = peak Reg-T margin, carried through). Sharpe is deliberately **not** the headline (flatters short-vol negative skew); skew/kurtosis/CVaR/Calmar/Sortino reported alongside it, at three horizons (daily aggregation hides the tail):
- **Primitives** (each unit-tested vs manual/scipy): `max_drawdown` (peak-to-trough of the cumsum level), `cvar` (expected shortfall = mean of worst ⌈αn⌉, k-smallest not quantile-interp so it's exact on tiny samples), `sortino` (downside-dev Sharpe, all-N denominator), `distribution_stats` (mean/std/Sharpe/skew/excess-kurt/CVaR/maxDD/Calmar/Sortino/win-rate).
- **Horizons:** daily (ppy 252), weekly (daily summed into calendar weeks, ppy 52), per-trade (10 positions' net PnL / capital base, **non-annualized**, thin-sample flagged). Arithmetic returns on a fixed base → horizons add. rf=0 (financing already in engine PnL).
- **Real numbers:** daily ann Sharpe **−1.70**, skew **+2.60**, excess-kurt **+15.2**, CVaR5% −0.49%, maxDD 2.79%, Calmar −2.57, Sortino −2.84. Weekly Sharpe −1.69. Per-trade skew **−0.83** (classic short-vol negative skew), win-rate 60%, worst −1.45%. Negative Sharpe is by construction (net<0 after costs); shape stats still describe the payoff.
- Output: `results/metrics.json` (PLAN deliverable, tracked, byte-stable — verified). Wired into `main.py --stage backtest` after `run_returns()`.
- `tests/test_metrics.py` (10): maxDD known-trough + monotone-zero, CVaR worst-fraction + small-sample, Sortino manual, shape stats vs scipy, Sharpe ann + per-trade non-ann, Calmar identity, real gate (3 horizon blocks finite, 10 trades, daily Sharpe<0, denominator carried). Suite: **628 passed**.

### Day 27 — Event PnL table + alpha isolation (Done)

**Status:** Completed. `src/backtest/alpha.py` — both deliverables merged into `results/metrics.json`:
- **Newey-West HAC regression** (`newey_west_ols`, Bartlett kernel, implemented here — no statsmodels): book net daily return ~ 1 + VRP factor.
  - **Factor construction (stated, SPEC):** daily PnL of ONE delta-hedged, **non-rolled** short ATM straddle on AAPL, entered first trade date → held to last path date, marked at mean short-leg IV, same capital base. Delta-hedging strips direction → pure gamma/theta = harvested vol premium. (Non-rolled → gamma fades off the fixed strike; documented.)
  - **Lags = ⌈median holding⌉ = 22** trading days (the ~monthly-hold overlap).
  - **Real result: beta −1.59 (NW t=−10.5), R²=0.53, corr −0.72** — book loads *negatively* on a short-vol factor ⇒ it's net **long-vol** (long legs longer-dated, bigger gamma). **alpha −0.00019/day, NW t=−0.95 → statistically ZERO.** After stripping vol beta, no residual edge — the disproof thesis, quantified.
- **Event PnL table:** top-5 |move| days in the trade window; only **2023-08-04 AAPL earnings gap (−4.80%)** exceeds 3% (`is_event`). Named events (Volmageddon/COVID/Aug-2024) are pre-2024/pre-window → reported OOS, no data. Each row carries book daily return, book margin, **ΔMargin** (procyclicality). **Honest nuance surfaced:** this net-long-vol book *gained* +1.36% on the 08-04 gap (long gamma) while margin *fell* — the opposite of the short-vol "tail hit twice"; the procyclical margin-up-as-equity-falls pattern shows on short-vol-dominated days (e.g. 06-22) instead. Note documents this rather than overclaiming.
- Merged into `metrics.json` under `alpha_regression` + `event_table` (byte-stable — verified). Wired into `main.py` after `run_metrics()`.
- `tests/test_alpha.py` (7): OLS vs lstsq, perfect-fit zero-SE, **NW lag-0 == White HC0** (independent recompute), **autocorrelation inflates HAC SE**, event-table flags the >3% move, real gates (VRP factor finite, |alpha t|<2 no-edge, 08-04 flagged, persisted). Suite: **635 passed**.

### Day 28 — Stats honesty (Done)

**Status:** Completed. `src/backtest/stats.py` — Sharpe uncertainty merged into `results/metrics.json` under `statistical_honesty`:
- **Sharpe + Newey-West SE/t** (`nw_mean_se`, Bartlett, lags = ⌈median holding⌉ = 22): the annualized Sharpe ∝ mean return, so its HAC t = mean/HAC-SE(mean). **Sharpe −1.70, NW SE 1.97, t = −0.86** (|t|<1 → insignificant).
- **Block-bootstrap 95% CI** (`block_bootstrap_sharpe`, moving-block length ⌈√T⌉=8, 2000 draws, seed 0 → byte-stable): **[−7.43, +2.00]** — spans zero. Sharpe not statistically distinguishable from zero.
- **IS-vs-OOS haircut** (chronological split-half): SR_is −5.03 → SR_oos +1.32, haircut −6.35. Negative haircut (OOS>IS) is noise at n=27/side — flagged, reported honestly.
- **Deflated Sharpe deliberately deferred** (`computed:false` + reason): DSR needs an honest multiple-testing N, complete only after v2 robustness sweeps (PLAN v2 Day 37). Computing DSR now with N=1 would understate it → recorded as deferred, not faked. Respects pre-registration.
- Merged into `metrics.json` (byte-stable — verified). Wired into `main.py` after `run_alpha()`.
- `tests/test_stats.py` (8): NW-SE lag-0 == √(γ₀/T), autocorrelation inflates HAC SE, Sharpe formula, HAC-t == mean/SE, bootstrap seeded+ordered+brackets-true-SR, IS/OOS split, real gate (Sharpe<0, finite CI, DSR deferred). Suite: **643 passed**.

### Day 29 — report.html + README (Next)

**Goal (PLAN):** single-page HTML report (surface + PnL attribution + return-distribution/event/stats) + README polish. Then Day 30 = final v1 ship.

See [PLAN.md](file:///c:/Users/ajiit/Desktop/Future/Projects/Options%20Trading/PLAN.md) for full Day 2–30 sequence.

---

## Key Files

| File | Purpose |
|------|---------|
| `SPEC.md` | Full project specification |
| `PLAN.md` | Day-by-day implementation plan |
| `HANDOFF.md` | This file — resume context |
| `notebooks/00_data_audit.py` | Day 1 data gate audit |
| `scripts/sha256_manifest.py` | Data immutability verification |
| `results/data_quality.json` | Audit output (generated) |
| `data/raw/manifest.json` | SHA256 hashes (generated) |

---

## Architecture Notes

- **Stack:** Python 3.11+, numpy, pandas, scipy, matplotlib, pyyaml
- **Reproducibility:** Fixed seed, pinned versions, `python main.py` regenerates everything
- **Module layout:** `src/{surface, backtest, greeks, utils}/`
- **Raw data:** `data/raw/` is immutable, SHA256 manifest
- **All results:** `results/` dir, single `metrics.json` source of truth

---

*Last updated: 2026-07-08 — Day 28 completed: Sharpe honesty → metrics.json (Sharpe −1.70, NW t=−0.86, bootstrap 95% CI [−7.43,+2.00] spans zero, IS/OOS haircut −6.35; Deflated Sharpe deferred to v2 per pre-registration). Suite: 643 passed.*

