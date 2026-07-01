"""
Download options data from DoltHub — date-by-date approach.
============================================================
Avoids DISTINCT/aggregate queries that timeout on large tables.
Generates business dates locally, queries each one.

Usage:
    python scripts/download_options.py
    python scripts/download_options.py --ticker AAPL --start 2023-01-02 --end 2023-03-31
"""

import argparse
import json
import sys
import time
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

DOLTHUB_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
ROWS_PER_QUERY = 200


def query_dolthub(sql: str, retries: int = 3, timeout: int = 45) -> dict:
    """Execute SQL query against DoltHub API."""
    url = f"{DOLTHUB_API}?q={urllib.parse.quote(sql)}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("query_execution_status") == "Success":
                return data
            else:
                msg = data.get("query_execution_message", "unknown")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                return data
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return {"query_execution_status": "Error", "rows": []}


def generate_business_dates(start: str, end: str) -> list[str]:
    """Generate weekday dates (Mon-Fri) between start and end."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    dates = []
    d = s
    while d <= e:
        if d.weekday() < 5:  # Mon=0, Fri=4
            dates.append(d.isoformat())
        d += timedelta(days=1)
    return dates


def download_date(ticker: str, query_date: str) -> list[dict]:
    """Download all rows for one date, paginating through."""
    all_rows = []
    offset = 0
    while True:
        sql = (
            f"SELECT * FROM option_chain "
            f"WHERE act_symbol = '{ticker}' AND date = '{query_date}' "
            f"ORDER BY expiration, strike, call_put "
            f"LIMIT {ROWS_PER_QUERY} OFFSET {offset}"
        )
        data = query_dolthub(sql)
        rows = data.get("rows", [])
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < ROWS_PER_QUERY:
            break
        offset += ROWS_PER_QUERY
        time.sleep(0.2)  # rate limit between pages
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Download options from DoltHub (date-by-date)")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2023-01-02")
    parser.add_argument("--end", type=str, default="2023-06-30")
    parser.add_argument("--max-dates", type=int, default=None,
                        help="Stop after N dates (for testing)")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    ticker = args.ticker.upper()

    print(f"Download: {ticker} [{args.start} -> {args.end}]")

    # Generate business dates locally (no DoltHub query needed)
    dates = generate_business_dates(args.start, args.end)
    if args.max_dates:
        dates = dates[:args.max_dates]
    print(f"Will try {len(dates)} business dates")

    all_rows = []
    dates_with_data = 0
    empty_streak = 0

    for i, d in enumerate(dates):
        rows = download_date(ticker, d)
        if rows:
            all_rows.extend(rows)
            dates_with_data += 1
            empty_streak = 0
        else:
            empty_streak += 1

        # Progress every 5 dates or on data
        if rows or (i + 1) % 5 == 0 or i == len(dates) - 1:
            print(f"  [{i+1}/{len(dates)}] {d}: {len(rows)} rows "
                  f"(total: {len(all_rows):,}, dates w/ data: {dates_with_data})")

        # If 30 consecutive empties, likely wrong date range
        if empty_streak >= 30:
            print(f"  30 consecutive empty dates. Stopping — check date range.")
            break

        time.sleep(0.5)  # polite rate limit

    if not all_rows:
        print("ERROR: No rows downloaded. Ticker may not exist in this DB.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # Ensure numeric types
    for col in ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path = DATA_RAW / f"{ticker.lower()}_options.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} rows -> {out_path}")
    print(f"Size: {out_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Dates with data: {dates_with_data}")

    # Quick sanity
    print(f"\nSchema: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Bid > 0: {(df['bid'] > 0).sum():,} / {len(df):,}")
    print(f"Ask > 0: {(df['ask'] > 0).sum():,} / {len(df):,}")
    has_bid_ask = (df["bid"] > 0) & (df["ask"] > 0) & (df["bid"] < df["ask"])
    print(f"Valid quotes (bid>0, ask>0, bid<ask): {has_bid_ask.sum():,} / {len(df):,} ({has_bid_ask.mean():.1%})")


if __name__ == "__main__":
    main()
