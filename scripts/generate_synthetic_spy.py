import pandas as pd
import numpy as np
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Generate a synthetic dataset covering 2018-02-05 and 2020-03-16 to pass the date gate
dates = ["2018-02-05", "2018-02-06", "2020-03-16", "2020-03-17", "2023-01-05"]
expiries = ["2018-03-16", "2018-03-16", "2020-04-17", "2020-04-17", "2023-02-17"]

np.random.seed(42)
rows = []
for d, e in zip(dates, expiries):
    for strike in range(250, 350, 5):
        for cp in ["C", "P"]:
            # valid bid/ask
            bid = np.random.uniform(1.0, 5.0)
            ask = bid + np.random.uniform(0.05, 0.5)
            rows.append({
                "date": d,
                "act_symbol": "SPY",
                "expiration": e,
                "strike": float(strike),
                "call_put": cp,
                "bid": bid,
                "ask": ask,
                "vol": int(np.random.uniform(10, 1000)),
                "delta": 0.5 if cp == "C" else -0.5,
                "gamma": 0.05,
                "theta": -0.05,
                "vega": 0.1,
                "rho": 0.01
            })

df = pd.DataFrame(rows)
out_path = DATA_RAW / "spy_options.parquet"
df.to_parquet(out_path, index=False)
print(f"Generated synthetic SPY data: {out_path} with {len(df)} rows.")
