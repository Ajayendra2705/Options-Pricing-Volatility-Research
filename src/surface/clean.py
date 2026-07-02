"""
Day 6 — Quote-cleaning pipeline.

Pure-function core (`clean_chain`) + CLI entry (`run_cleaning`) that:
  1. loads the raw chain from data/raw/,
  2. normalizes columns (date, expiry, strike, option_type C/P, bid, ask),
  3. applies quote-quality filters:
       missing   bid or ask NaN
       zero_bid  bid <= 0
       crossed   bid >= ask
       wide      (ask - bid) / mid > max_spread_pct
       stale     identical bid AND ask as the same contract's previous
                 trading date (first observation never stale)
  4. adds mid price and T (ACT/365 year fraction expiry - quote date),
  5. writes cleaned chain -> data/processed/chain_clean.parquet,
  6. appends real drop counts -> results/data_quality.json under "cleaning".

Filters are applied as a union: a row is dropped if ANY filter flags it;
per-filter counts are reported pre-union (a row can trip several).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

MAX_SPREAD_PCT = 0.50
DAYS_PER_YEAR = 365.0

# canonical column -> accepted source names (lowercased)
COLUMN_ALIASES = {
    "date": ["date", "quote_date", "trade_date", "quotedate", "datadate"],
    "expiry": ["expiry", "expiration", "expire_date", "exdate", "expiration_date"],
    "strike": ["strike", "strike_price"],
    "option_type": ["option_type", "call_put", "cp_flag", "type", "put_call"],
    "bid": ["bid", "best_bid", "bid_price"],
    "ask": ["ask", "best_ask", "ask_price", "best_offer"],
    "volume": ["volume", "vol"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to canonical names; parse dates; map option_type to C/P."""
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                rename[lower[a]] = canon
                break
    out = df.rename(columns=rename)

    missing = {"date", "expiry", "strike", "option_type", "bid", "ask"} - set(out.columns)
    if missing:
        raise ValueError(f"Raw chain missing required columns: {sorted(missing)}")

    out["date"] = pd.to_datetime(out["date"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out["option_type"] = (
        out["option_type"].astype(str).str.strip().str.upper().str[0]  # Call/Put/C/P -> C/P
    )
    bad_type = ~out["option_type"].isin(["C", "P"])
    if bad_type.any():
        raise ValueError(f"Unrecognized option_type values: {out.loc[bad_type, 'option_type'].unique()}")
    return out


def flag_quality(df: pd.DataFrame, max_spread_pct: float = MAX_SPREAD_PCT) -> pd.DataFrame:
    """Return boolean flag DataFrame (same index): missing, zero_bid, crossed, wide, stale."""
    flags = pd.DataFrame(index=df.index)
    flags["missing"] = df["bid"].isna() | df["ask"].isna()
    flags["zero_bid"] = df["bid"] <= 0
    flags["crossed"] = df["bid"] >= df["ask"]

    mid = (df["bid"] + df["ask"]) / 2.0
    spread = df["ask"] - df["bid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_pct = np.where(mid > 0, spread / mid, np.inf)
    flags["wide"] = spread_pct > max_spread_pct

    # stale: same contract, previous trading date, identical bid AND ask
    contract = ["expiry", "strike", "option_type"]
    srt = df.sort_values("date")
    grp = srt.groupby(contract, sort=False)
    same_bid = grp["bid"].shift() == srt["bid"]
    same_ask = grp["ask"].shift() == srt["ask"]
    flags["stale"] = (same_bid & same_ask).reindex(df.index).fillna(False)

    # NaN comparisons already yield False; make dtype explicit
    return flags.fillna(False).astype(bool)


def clean_chain(
    df: pd.DataFrame, max_spread_pct: float = MAX_SPREAD_PCT
) -> tuple[pd.DataFrame, dict]:
    """Normalize + filter a raw chain. Returns (clean_df, report).

    clean_df gains: mid, T (ACT/365 year fraction). report holds per-filter
    counts (pre-union), union drop count and drop rate.
    """
    df = normalize_columns(df)
    flags = flag_quality(df, max_spread_pct)
    drop = flags.any(axis=1)

    clean = df[~drop].copy()
    clean["mid"] = (clean["bid"] + clean["ask"]) / 2.0
    clean["T"] = (clean["expiry"] - clean["date"]).dt.days / DAYS_PER_YEAR
    nonpos_T = clean["T"] <= 0
    n_nonpos_T = int(nonpos_T.sum())
    clean = clean[~nonpos_T].reset_index(drop=True)

    n_total = len(df)
    n_dropped = int(drop.sum()) + n_nonpos_T
    report = {
        "total_rows": n_total,
        "max_spread_pct": max_spread_pct,
        "filters": {k: int(flags[k].sum()) for k in flags.columns},
        "expired_nonpos_T": n_nonpos_T,
        "total_dropped": n_dropped,
        "total_clean": len(clean),
        "drop_rate": round(n_dropped / n_total, 4) if n_total else 0.0,
    }
    return clean, report


def run_cleaning(
    raw_dir: Path = RAW_DIR,
    out_path: Path | None = None,
    max_spread_pct: float = MAX_SPREAD_PCT,
) -> Path:
    """Load raw parquet/csv chain(s), clean, persist parquet + drop counts."""
    files = sorted(
        f for f in raw_dir.rglob("*") if f.is_file() and f.suffix in (".parquet", ".csv")
        and f.name != "manifest.json"
    )
    if not files:
        raise FileNotFoundError(f"No raw data files in {raw_dir}")
    frames = [pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f) for f in files]
    raw = pd.concat(frames, ignore_index=True)

    clean, report = clean_chain(raw, max_spread_pct)

    out_path = out_path or PROCESSED_DIR / "chain_clean.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)

    # append real drop counts to results/data_quality.json
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dq_path = RESULTS_DIR / "data_quality.json"
    dq = json.loads(dq_path.read_text()) if dq_path.exists() else {}
    # no timestamp: keeps the tracked json byte-stable across identical reruns
    dq["cleaning"] = {
        "source_files": [f.name for f in files],
        "output": str(out_path.relative_to(PROJECT_ROOT)),
        **report,
    }
    dq_path.write_text(json.dumps(dq, indent=2, default=str))

    print(
        f"clean_chain: {report['total_rows']} raw -> {report['total_clean']} clean "
        f"(drop rate {report['drop_rate']:.1%}); filters {report['filters']}"
    )
    print(f"-> {out_path}")
    print(f"-> {dq_path}")
    return out_path


if __name__ == "__main__":
    run_cleaning()
