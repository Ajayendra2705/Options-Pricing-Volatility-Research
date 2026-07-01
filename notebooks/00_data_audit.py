"""
Day 1 — DATA GATE: SPY Options Data Audit
==========================================
Load raw dataset, verify two-sided quotes exist, run quote-quality filters,
compute drop rates, output results/data_quality.json.

Gate decision: real quotes present + drop rate < ~40% → PROCEED.

Usage:
    python notebooks/00_data_audit.py
    python notebooks/00_data_audit.py --data-dir data/raw/custom_path
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# Quote-quality thresholds
MAX_SPREAD_PCT = 0.50  # drop if spread > 50% of mid
MIN_BID = 0.0          # drop zero-bid
GATE_DROP_THRESHOLD = 0.40  # proceed if total drop rate < 40%


def find_data_files(data_dir: Path) -> list[Path]:
    """Find parquet/csv files in data dir."""
    exts = {".parquet", ".csv", ".parquet.gz"}
    files = []
    for f in sorted(data_dir.rglob("*")):
        if f.is_file() and f.suffix in exts:
            files.append(f)
        elif f.is_file() and ".parquet" in f.name:
            files.append(f)
    return files


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load all data files from data dir into single DataFrame."""
    files = find_data_files(data_dir)
    if not files:
        print(f"ERROR: No data files found in {data_dir}")
        print("Download options dataset -> data/raw/")
        print("  Option A: https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2024")
        print("  Option B: Any SPY/SPX options dataset w/ bid/ask columns")
        sys.exit(1)

    print(f"Found {len(files)} data file(s):")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

    dfs = []
    for f in files:
        if f.suffix == ".csv":
            dfs.append(pd.read_csv(f, low_memory=False))
        else:
            dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(df):,}")
    return df


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect column mappings. SPY EOD datasets vary in naming.
    Returns dict mapping canonical names → actual column names.
    """
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    # Date column
    for candidate in ["quote_date", "date", "datadate", "trade_date", "quotedate"]:
        if candidate in cols:
            mapping["date"] = cols[candidate]
            break

    # Expiry
    for candidate in ["expire_date", "expiration", "exdate", "expiry", "expiredate",
                       "expiration_date"]:
        if candidate in cols:
            mapping["expiry"] = cols[candidate]
            break

    # Strike
    for candidate in ["strike", "strike_price", "strikeprice"]:
        if candidate in cols:
            mapping["strike"] = cols[candidate]
            break

    # Option type
    for candidate in ["option_type", "cp_flag", "type", "call_put", "optiontype",
                       "putcall", "put_call"]:
        if candidate in cols:
            mapping["option_type"] = cols[candidate]
            break

    # Call bid/ask (separate columns pattern)
    for candidate in ["c_bid", "call_bid"]:
        if candidate in cols:
            mapping["c_bid"] = cols[candidate]
            break

    for candidate in ["c_ask", "call_ask"]:
        if candidate in cols:
            mapping["c_ask"] = cols[candidate]
            break

    # Put bid/ask (separate columns pattern)
    for candidate in ["p_bid", "put_bid"]:
        if candidate in cols:
            mapping["p_bid"] = cols[candidate]
            break

    for candidate in ["p_ask", "put_ask"]:
        if candidate in cols:
            mapping["p_ask"] = cols[candidate]
            break

    # Generic bid/ask (single-row-per-option pattern)
    for candidate in ["bid", "best_bid", "bid_price", "bidprice"]:
        if candidate in cols:
            mapping["bid"] = cols[candidate]
            break

    for candidate in ["ask", "best_ask", "ask_price", "askprice", "best_offer"]:
        if candidate in cols:
            mapping["ask"] = cols[candidate]
            break

    # Underlying price
    for candidate in ["underlying_last", "spot", "underlying", "undprice",
                       "stock_price", "close", "adj_close"]:
        if candidate in cols:
            mapping["underlying"] = cols[candidate]
            break

    # Volume / open interest
    for candidate in ["volume", "vol", "c_volume", "p_volume"]:
        if candidate in cols:
            mapping["volume"] = cols[candidate]
            break

    for candidate in ["open_interest", "oi", "openinterest", "c_open_interest"]:
        if candidate in cols:
            mapping["open_interest"] = cols[candidate]
            break

    # IV (pre-computed)
    for candidate in ["implied_volatility", "iv", "c_iv", "impl_vol",
                       "impliedvolatility"]:
        if candidate in cols:
            mapping["iv"] = cols[candidate]
            break

    return mapping


def normalize_to_long(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Normalize to long format: one row per option with [date, expiry, strike,
    option_type, bid, ask, ...].

    Handles two patterns:
      A) Already long: has 'bid', 'ask', 'option_type' columns
      B) Wide: has 'c_bid', 'c_ask', 'p_bid', 'p_ask' on same row
    """
    has_generic_bid = "bid" in mapping
    has_split_bid = "c_bid" in mapping and "p_bid" in mapping

    if has_generic_bid:
        # Pattern A — already long format
        rename = {}
        if "date" in mapping:
            rename[mapping["date"]] = "date"
        if "expiry" in mapping:
            rename[mapping["expiry"]] = "expiry"
        if "strike" in mapping:
            rename[mapping["strike"]] = "strike"
        if "option_type" in mapping:
            rename[mapping["option_type"]] = "option_type"
        rename[mapping["bid"]] = "bid"
        rename[mapping["ask"]] = "ask"
        if "underlying" in mapping:
            rename[mapping["underlying"]] = "underlying"
        if "volume" in mapping:
            rename[mapping["volume"]] = "volume"
        if "open_interest" in mapping:
            rename[mapping["open_interest"]] = "open_interest"
        if "iv" in mapping:
            rename[mapping["iv"]] = "iv"

        out = df.rename(columns=rename)
        return out

    elif has_split_bid:
        # Pattern B — wide format, melt to long
        base_cols = []
        rename = {}
        if "date" in mapping:
            rename[mapping["date"]] = "date"
            base_cols.append("date")
        if "expiry" in mapping:
            rename[mapping["expiry"]] = "expiry"
            base_cols.append("expiry")
        if "strike" in mapping:
            rename[mapping["strike"]] = "strike"
            base_cols.append("strike")
        if "underlying" in mapping:
            rename[mapping["underlying"]] = "underlying"
            base_cols.append("underlying")

        df_r = df.rename(columns=rename)

        # Build call rows
        calls = df_r[base_cols].copy()
        calls["option_type"] = "C"
        calls["bid"] = df[mapping["c_bid"]]
        calls["ask"] = df[mapping["c_ask"]]
        # Try to get call-specific volume/OI/IV
        for col_key, target in [("volume", "volume"), ("open_interest", "open_interest"),
                                 ("iv", "iv")]:
            if col_key in mapping:
                calls[target] = df[mapping[col_key]]

        # Build put rows
        puts = df_r[base_cols].copy()
        puts["option_type"] = "P"
        puts["bid"] = df[mapping["p_bid"]]
        puts["ask"] = df[mapping["p_ask"]]
        for col_key, target in [("volume", "volume"), ("open_interest", "open_interest"),
                                 ("iv", "iv")]:
            if col_key in mapping:
                puts[target] = df[mapping[col_key]]

        out = pd.concat([calls, puts], ignore_index=True)
        return out

    else:
        print("ERROR: Cannot find bid/ask columns in dataset.")
        print(f"  Detected mapping: {mapping}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)


def run_quality_filters(df: pd.DataFrame) -> dict:
    """
    Apply quote-quality filters. Returns dict with counts + filtered df.
    Filters:
      1. Missing bid/ask (NaN)
      2. Zero-bid (bid <= 0)
      3. Crossed quotes (bid >= ask)
      4. Wide spread (spread > MAX_SPREAD_PCT of mid)
    """
    n_total = len(df)
    results = {"total_rows": n_total, "filters": {}}

    # 1. Missing bid/ask
    mask_missing = df["bid"].isna() | df["ask"].isna()
    n_missing = int(mask_missing.sum())
    results["filters"]["missing_bid_ask"] = n_missing

    # Work with non-missing from here
    df_valid = df[~mask_missing].copy()

    # 2. Zero-bid
    mask_zero = df_valid["bid"] <= MIN_BID
    n_zero = int(mask_zero.sum())
    results["filters"]["zero_bid"] = n_zero

    # 3. Crossed quotes
    mask_crossed = df_valid["bid"] >= df_valid["ask"]
    n_crossed = int(mask_crossed.sum())
    results["filters"]["crossed_quotes"] = n_crossed

    # 4. Wide spread
    mid = (df_valid["bid"] + df_valid["ask"]) / 2.0
    spread = df_valid["ask"] - df_valid["bid"]
    # Avoid div-by-zero on mid == 0
    spread_pct = np.where(mid > 0, spread / mid, np.inf)
    mask_wide = spread_pct > MAX_SPREAD_PCT
    n_wide = int(mask_wide.sum())
    results["filters"]["wide_spread"] = n_wide

    # Combined filter (union of all bad)
    mask_any_bad = mask_missing.reindex(df_valid.index, fill_value=False) | mask_zero | mask_crossed | mask_wide
    n_dropped = int(mask_any_bad.sum())
    n_clean = len(df_valid) - n_dropped
    drop_rate = n_dropped / n_total if n_total > 0 else 0.0

    results["total_dropped"] = n_dropped + n_missing  # include missing too
    results["total_clean"] = n_clean
    results["drop_rate"] = round(drop_rate, 4)

    # Clean df
    df_clean = df_valid[~mask_any_bad].copy()
    results["_df_clean"] = df_clean

    return results


def coverage_scan(df: pd.DataFrame) -> dict:
    """Scan coverage: date range, expiry counts, gap dates."""
    info = {}

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        dates = dates.dropna()
        if len(dates) > 0:
            info["date_range"] = [str(dates.min().date()), str(dates.max().date())]
            unique_dates = sorted(dates.dt.date.unique())
            info["unique_dates_count"] = len(unique_dates)

            # Find gaps > 3 calendar days (excl weekends)
            if len(unique_dates) > 1:
                gaps = []
                for i in range(1, len(unique_dates)):
                    delta = (unique_dates[i] - unique_dates[i - 1]).days
                    if delta > 4:  # more than a long weekend
                        gaps.append({
                            "from": str(unique_dates[i - 1]),
                            "to": str(unique_dates[i]),
                            "gap_days": delta
                        })
                info["date_gaps_gt_4d"] = gaps[:20]  # cap at 20

    if "expiry" in df.columns:
        expiries = pd.to_datetime(df["expiry"], errors="coerce").dropna()
        info["unique_expiries_count"] = int(expiries.dt.date.nunique())

    if "strike" in df.columns:
        info["strike_range"] = [float(df["strike"].min()), float(df["strike"].max())]
        info["unique_strikes_count"] = int(df["strike"].nunique())

    if "option_type" in df.columns:
        info["option_type_counts"] = df["option_type"].value_counts().to_dict()

    return info


def print_gate_decision(quality: dict, coverage: dict):
    """Print the gate decision."""
    drop_rate = quality["drop_rate"]
    has_quotes = quality["total_clean"] > 0

    print("\n" + "=" * 60)
    print("  DATA GATE DECISION")
    print("=" * 60)
    print(f"  Total rows:       {quality['total_rows']:>12,}")
    print(f"  Clean rows:       {quality['total_clean']:>12,}")
    print(f"  Dropped rows:     {quality['total_dropped']:>12,}")
    print(f"  Drop rate:        {drop_rate:>11.1%}")
    print()

    for fname, count in quality["filters"].items():
        pct = count / quality["total_rows"] * 100 if quality["total_rows"] > 0 else 0
        print(f"    {fname:<25s} {count:>10,}  ({pct:.1f}%)")

    print()
    if "date_range" in coverage:
        print(f"  Date range:       {coverage['date_range'][0]} -> {coverage['date_range'][1]}")
    if "unique_dates_count" in coverage:
        print(f"  Trading dates:    {coverage['unique_dates_count']:>12,}")
    if "unique_expiries_count" in coverage:
        print(f"  Unique expiries:  {coverage['unique_expiries_count']:>12,}")
    if "unique_strikes_count" in coverage:
        print(f"  Unique strikes:   {coverage['unique_strikes_count']:>12,}")

    print()
    if has_quotes and drop_rate < GATE_DROP_THRESHOLD:
        print("  [PASS] GATE: PROCEED")
        print(f"     Real two-sided quotes present, drop rate {drop_rate:.1%} < {GATE_DROP_THRESHOLD:.0%} threshold.")
    elif has_quotes and drop_rate >= GATE_DROP_THRESHOLD:
        print("  [WARN] GATE: MARGINAL")
        print(f"     Real quotes present but drop rate {drop_rate:.1%} >= {GATE_DROP_THRESHOLD:.0%}.")
        print("     Consider tightening filters or switching to cleaner date window.")
    else:
        print("  [FAIL] GATE: FAIL")
        print("     No usable quotes after filtering. Switch data source.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Day 1 Data Audit — SPY Options")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Path to raw data directory")
    parser.add_argument("--spread-pct", type=float, default=MAX_SPREAD_PCT,
                        help=f"Max spread as fraction of mid (default {MAX_SPREAD_PCT})")
    args = parser.parse_args()

    spread_pct = args.spread_pct

    data_dir = Path(args.data_dir)
    print(f"Data audit -- scanning: {data_dir}")
    print(f"Spread threshold: {spread_pct:.0%}")
    print()

    # Load
    df_raw = load_data(data_dir)

    # Schema inspect
    print(f"\nColumns ({len(df_raw.columns)}): {list(df_raw.columns)}")
    print(f"\nDtypes:\n{df_raw.dtypes}")
    print(f"\nSample (first 3 rows):\n{df_raw.head(3).to_string()}")

    # Detect columns
    mapping = detect_columns(df_raw)
    print(f"\nDetected column mapping:")
    for k, v in mapping.items():
        print(f"  {k:<20s} -> {v}")

    # Normalize to long format
    print("\nNormalizing to long format...")
    df_long = normalize_to_long(df_raw, mapping)
    print(f"Long-format rows: {len(df_long):,}")

    # Ensure numeric bid/ask
    df_long["bid"] = pd.to_numeric(df_long["bid"], errors="coerce")
    df_long["ask"] = pd.to_numeric(df_long["ask"], errors="coerce")

    # Quality filters
    print("\nRunning quote-quality filters...")
    quality = run_quality_filters(df_long)
    df_clean = quality.pop("_df_clean")

    # Coverage scan
    coverage = coverage_scan(df_long)
    coverage_clean = coverage_scan(df_clean)

    # Gate decision
    print_gate_decision(quality, coverage)

    # Write results/data_quality.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "audit_timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "spread_threshold": MAX_SPREAD_PCT,
        "raw_schema": list(df_raw.columns),
        "column_mapping": mapping,
        "quality": quality,
        "coverage_raw": coverage,
        "coverage_clean": coverage_clean,
        "gate_decision": (
            "PROCEED" if quality["total_clean"] > 0 and quality["drop_rate"] < GATE_DROP_THRESHOLD
            else "MARGINAL" if quality["total_clean"] > 0
            else "FAIL"
        ),
    }

    out_path = RESULTS_DIR / "data_quality.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n-> Wrote {out_path}")


if __name__ == "__main__":
    main()
