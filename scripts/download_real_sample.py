import pandas as pd
import sys
from pathlib import Path

# Add scripts dir to path to import download_options
sys.path.append(str(Path(__file__).parent))
from download_options import download_date, DATA_RAW

def main():
    ticker = "AAPL"
    # Specific dates to cover key events (Feb 2018, Mar 2020)
    dates = ["2018-02-05", "2018-02-06", "2020-03-16", "2020-03-17", "2023-01-05"]
    
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for d in dates:
        print(f"Downloading {d}...")
        rows = download_date(ticker, d)
        all_rows.extend(rows)
        print(f"  Got {len(rows)} rows.")

    if not all_rows:
        print("ERROR: No rows downloaded.")
        return

    df = pd.DataFrame(all_rows)
    for col in ["strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path = DATA_RAW / f"{ticker.lower()}_options.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df):,} rows -> {out_path}")

if __name__ == "__main__":
    main()
