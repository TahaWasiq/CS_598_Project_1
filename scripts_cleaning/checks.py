#!/usr/bin/env python3
"""
checks.py — quick sanity summary for each split.
Reads : data/processed/h{1,7}_{train,val,test}.csv
Writes: prints summary to stdout
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def summarize(name: str):
    p = PROCESSED / name
    df = pd.read_csv(p, parse_dates=["date"])
    target = [c for c in df.columns if c.startswith("y_btc_close_t+")][0]
    print(name)
    print(f"rows: {len(df)} cols: {df.shape[1]}")
    print(f"span: {df['date'].min().date()} → {df['date'].max().date()}")
    print("top NA:\n", df.isna().sum().sort_values(ascending=False).head(8))
    print("target describe:\n", df[target].describe(), "\n")

def main():
    for h in (1, 7):
        for split in ("train", "val", "test"):
            summarize(f"h{h}_vif_{split}.csv")

if __name__ == "__main__":
    main()
