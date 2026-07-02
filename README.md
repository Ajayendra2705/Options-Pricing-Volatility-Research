# Options Vol-Arb — Implied-Vol Surface & Delta-Hedged Backtest

> Arbitrage-free implied-volatility surface (SVI) + delta-hedged vol-arb backtest
> on equity options. Thesis: **disprove** exploitable VRP on liquid index after
> costs/margin/discrete-hedging, proven with reconciling Greeks attribution and
> tail-honest stats.

## Quick start

```bash
pip install -r requirements.txt
python main.py                    # regenerates all results from raw data
python main.py --stage clean      # cleaning only
python main.py --stage surface    # Part 1: forwards -> IVs -> SVI -> no-arb -> surface + QC
pytest                            # run test suite
```

Part 1 (vol surface) is locked: a full `python main.py --stage all` rerun
reproduces every processed parquet and results json bit-identically
(SHA256-verified).

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
