"""
Download options data from DoltHub (no auth required).
=====================================================
Uses DoltHub SQL API to pull option_chain data for a given ticker.
Paginates by date to avoid timeouts on large queries.

Usage:
    python scripts/download_dolthub.py
    python scripts/download_dolthub.py --ticker AAPL --start 2019-01-01 --end 2024-12-31
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

# DoltHub has S&P 500 individual stocks. SPY/SPX not present.
# AAPL is deeply liquid — same pipeline validation.
DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2019-01-02"
DEFAULT_END = "2024-06-30"
ROWS_PER_QUERY = 200  # DoltHub API limit


def query_dolthub(sql: str, retries: int = 3) -> dict:
    """Execute SQL query against DoltHub API."""
    url = f"{DOLTHUB_API}?q={urllib.parse.quote(sql)}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("query_execution_status") == "Success":
                return data
            else:
                print(f"  Query error: {data.get('query_execution_message')}")
                return data
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt + 1}/{retries}), retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Error: {e} (attempt {attempt + 1}/{retries})")
            time.sleep(2 ** attempt)
    return {"query_execution_status": "Error", "rows": []}


def get_date_range(ticker: str) -> tuple[str, str]:
    """Get min/max date for ticker."""
    sql = f"SELECT MIN(date) as min_d, MAX(date) as max_d FROM option_chain WHERE act_symbol = '{ticker}' LIMIT 1"
    data = query_dolthub(sql)
    if data.get("rows"):
        row = data["rows"][0]
        return row.get("min_d", ""), row.get("max_d", "")
    return "", ""


def download_by_date(ticker: str, query_date: str) -> list[dict]:
    """Download all option rows for a specific date and ticker."""
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
    return all_rows


def get_trading_dates(ticker: str, start: str, end: str) -> list[str]:
    """Get list of distinct trading dates for ticker in range."""
    all_dates = []
    # Paginate through dates
    sql = (
        f"SELECT DISTINCT date FROM option_chain "
        f"WHERE act_symbol = '{ticker}' AND date >= '{start}' AND date <= '{end}' "
        f"ORDER BY date"
    )
    data = query_dolthub(sql)
    if data.get("rows"):
        all_dates = [r["date"] for r in data["rows"]]
    return all_dates


def main():
    parser = argparse.ArgumentParser(description="Download options from DoltHub")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER,
                        help=f"Ticker symbol (default: {DEFAULT_TICKER})")
    parser.add_argument("--start", type=str, default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end", type=str, default=DEFAULT_END,
                        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})")
    parser.add_argument("--sample-dates", type=int, default=None,
                        help="Only download N evenly-spaced dates (for quick test)")
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    ticker = args.ticker.upper()

    print(f"DoltHub download: {ticker} [{args.start} -> {args.end}]")
    print(f"Output: {DATA_RAW}")

    # Get trading dates
    print("\nFetching trading dates...")
    dates = get_trading_dates(ticker, args.start, args.end)
    if not dates:
        print(f"ERROR: No data found for {ticker} in [{args.start}, {args.end}]")
        print("Available tickers (sample): A, AAL, AAPL, ABBV, ABT, ACN, ADBE, ...")
        sys.exit(1)

    print(f"Found {len(dates)} trading dates")

    # Sample if requested
    if args.sample_dates and args.sample_dates < len(dates):
        step = max(1, len(dates) // args.sample_dates)
        dates = dates[::step][:args.sample_dates]
        print(f"Sampled down to {len(dates)} dates")

    # Download date by date
    all_rows = []
    for i, d in enumerate(dates):
        rows = download_by_date(ticker, d)
        all_rows.extend(rows)
        if (i + 1) % 10 == 0 or i == 0 or i == len(dates) - 1:
            print(f"  [{i+1}/{len(dates)}] {d}: {len(rows)} rows (total: {len(all_rows):,})")
        # Polite rate limit
        time.sleep(0.3)

    if not all_rows:
        print("ERROR: No rows downloaded")
        sys.exit(1)

    # Convert to DataFrame + save as parquet
    df = pd.DataFrame(all_rows)
    out_path = DATA_RAW / f"{ticker.lower()}_options.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n✅ Saved {len(df):,} rows → {out_path}")
    print(f"   Size: {out_path.stat().st_size / (1024*1024):.1f} MB")

    # Also save as CSV for inspection
    csv_path = DATA_RAW / f"{ticker.lower()}_options_sample.csv"
    df.head(100).to_csv(csv_path, index=False)
    print(f"   Sample CSV (100 rows) → {csv_path}")


if __name__ == "__main__":
    main()
