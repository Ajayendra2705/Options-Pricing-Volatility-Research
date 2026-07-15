"""
Download daily OHLCV for the underlying from DoltHub (no auth).
===============================================================
Source: post-no-preference/stocks, table `ohlcv` — same provider as the
Day-1 options chain, so quote dates line up. Close 2023-06-02 = 180.95
matches the external anchor used in the Day-7 forward verification.

Usage:
    python scripts/download_ohlc.py
    python scripts/download_ohlc.py --ticker AAPL --start 2022-01-01 --end 2023-06-30
"""

import argparse
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

DOLTHUB_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/stocks/master"
ROWS_PER_QUERY = 200


def query(sql: str, retries: int = 8) -> list[dict]:
    url = f"{DOLTHUB_API}?q={urllib.parse.quote(sql)}"
    last = None
    for attempt in range(retries):
        if attempt:
            time.sleep(5 * attempt)                  # API rate-limits bursts
        resp = requests.get(url, timeout=90)
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}"
            continue
        data = resp.json()
        if data.get("query_execution_status") == "Success":
            return data["rows"]
        last = data.get("query_execution_message")   # e.g. server-side timeout
    raise RuntimeError(last)


def _windows(start: str, end: str, chunk_days: int):
    """Yield (win_start, win_end, cache_key) BETWEEN windows.

    chunk_days == 0 → calendar-month windows (original behaviour; fine for AAPL).
    chunk_days  > 0 → fixed N-day windows. SPY's ohlcv rows are far denser to
    scan, so a month-sized BETWEEN blows the API's server-side deadline
    ("context deadline exceeded" → status Error, partial rows); ~7-day windows
    complete in ~1.5s with status Success. Verified 2026-07-15.
    """
    if chunk_days <= 0:
        for m in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
            m_end = min(m + pd.offsets.MonthEnd(0), pd.Timestamp(end))
            yield m, m_end, f"{m:%Y-%m}"
    else:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        w = s
        while w <= e:
            w_end = min(w + pd.Timedelta(days=chunk_days - 1), e)
            yield w, w_end, f"{w:%Y-%m-%d}"
            w = w_end + pd.Timedelta(days=1)


def download(ticker: str, start: str, end: str, chunk_days: int = 0) -> pd.DataFrame:
    # Windowed BETWEEN queries: wide ranges hit the API's server-side query
    # deadline. Each window is cached to disk immediately so an interrupted run
    # resumes instead of restarting (the API throttles hard).
    cache_dir = PROJECT_ROOT / "data" / "processed" / "ohlc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for w, w_end, key in _windows(start, end, chunk_days):
        cache = cache_dir / f"{ticker.lower()}_{key}.json"
        if cache.exists():
            batch = pd.read_json(cache, convert_dates=False, dtype=False).to_dict("records")
            print(f"  {w.date()} .. {w_end.date()}  (cached, {len(batch)})")
        else:
            sql = (f"SELECT date, open, high, low, close, volume FROM ohlcv "
                   f"WHERE act_symbol='{ticker}' "
                   f"AND date BETWEEN '{w.date()}' AND '{w_end.date()}'")
            batch = query(sql)
            pd.DataFrame(batch).to_json(cache, orient="records")
            print(f"  {w.date()} .. {w_end.date()}  (+{len(batch)})")
            time.sleep(1.5)                          # stay under the rate limit
        rows.extend(batch)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype("int64")
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    # sanity: positive prices, high/low envelope
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9).all()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-06-30")
    ap.add_argument("--chunk-days", type=int, default=0,
                    help="Fixed N-day query windows (0=monthly). Use 7 for SPY: "
                         "month windows hit the API deadline on the dense ohlcv scan.")
    ap.add_argument("--out-dir", default=str(DATA_RAW),
                    help="Output dir for <ticker>_ohlc.parquet (default data/raw).")
    args = ap.parse_args()

    df = download(args.ticker, args.start, args.end, args.chunk_days)
    out = Path(args.out_dir) / f"{args.ticker.lower()}_ohlc.parquet"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} rows {df['date'].min().date()} .. {df['date'].max().date()} -> {out}")
    print("Now run: python scripts/sha256_manifest.py   (update the manifest)")


if __name__ == "__main__":
    main()
