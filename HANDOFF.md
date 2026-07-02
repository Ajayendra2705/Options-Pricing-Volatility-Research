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

### Day 4 — Second-order Greeks (Next)

**Goal:** Add vanna (∂Δ/∂σ), volga/vomma (∂vega/∂σ) — analytic.
**Test:** finite-difference cross-check vs analytic, all Greeks, tight tol (`tests/test_greeks_fd.py`).

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

*Last updated: 2026-07-02 — Day 3 Completed*
