# Options Vol-Arb — Implied-Vol Surface & Delta-Hedged Backtest

> Arbitrage-free implied-volatility surface (SVI) + delta-hedged vol-arb backtest
> on equity options. Thesis: **disprove** exploitable VRP on liquid index after
> costs/margin/discrete-hedging, proven with reconciling Greeks attribution and
> tail-honest stats.

## Quick start

```bash
pip install -r requirements.txt
python main.py          # regenerates all results from raw data
pytest                  # run test suite
```

## Project structure

```
data/raw/               # Immutable raw data (SHA256 manifest)
data/processed/         # Cleaned/filtered data (generated)
src/
  greeks/               # BS pricing, IV inversion, Greeks
  surface/              # SVI calibration, no-arb constraints
  backtest/             # Delta-hedge engine, PnL attribution
  utils/                # Fixed seed, config, helpers
notebooks/              # Data audit, exploration
results/                # Generated metrics, plots, reports
tests/                  # pytest suite
config/                 # Strategy configs (YAML)
scripts/                # Data download, manifest tools
```

## Reproducibility

- Fixed random seed via `src/utils/seed.py`
- Pinned dependencies in `requirements.txt`
- Raw data locked via SHA256 manifest (`data/raw/manifest.json`)
- `python main.py` regenerates everything from scratch

## Status

See [HANDOFF.md](HANDOFF.md) for current progress and [PLAN.md](PLAN.md) for the
day-by-day implementation plan.
