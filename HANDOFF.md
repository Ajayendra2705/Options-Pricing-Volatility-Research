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

### Day 9 — Raw SVI calibration, one slice (Next)

**Goal:** `src/surface/svi.py` — raw SVI (a,b,ρ,m,σ), least-squares fit one maturity. Fitted smile vs market scatter, per-slice RMSE printed.

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

- **Stack:** Python 3.11+, numpy, pandas, scipy, matplotlib
- **Reproducibility:** Fixed seed, pinned versions, `python main.py` regenerates everything
- **Module layout:** `src/{surface, backtest, greeks, utils}/`
- **Raw data:** `data/raw/` is immutable, SHA256 manifest
- **All results:** `results/` dir, single `metrics.json` source of truth

---

*Last updated: 2026-07-02 — Day 8 Completed*
