#!/usr/bin/env python3
"""
clean.py — read one bitcoin.csv, normalize columns, compute daily return,
and save a tidy aligned daily frame for feature building.
Reads : data/raw/bitcoin.csv
Writes: data/interim/bitcoin_aligned.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

def read_bitcoin(path: Path) -> pd.DataFrame:
    # load then select and rename the key series we’ll use downstream
    df = pd.read_csv(path)
    required = ["Date", "btc_market_price", "btc_market_cap", "btc_trade_volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in {path.name}: {missing}")

    df = df[required].rename(
        columns={
            "Date": "date",
            "btc_market_price": "btc_close",     # treat market price as “close”
            "btc_market_cap": "btc_mktcap",
            "btc_trade_volume": "btc_volume",    # use on-chain trade volume
        }
    )

    #parse date and sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["btc_close", "btc_mktcap", "btc_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    #remove duplicate dates, keep last
    df = df.drop_duplicates(subset=["date"], keep="last")

    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_idx)
    df.index.name = "date"

    #only forward-fill very short gaps
    df[["btc_close", "btc_mktcap", "btc_volume"]] = df[
        ["btc_close", "btc_mktcap", "btc_volume"]
    ].fillna(method="ffill", limit=3)

    #daily % return on btc_close
    df["btc_ret1"] = df["btc_close"].pct_change()

    # final
    df = df.reset_index().rename(columns={"index": "date"})
    return df[["date", "btc_close", "btc_volume", "btc_mktcap", "btc_ret1"]]

def main():
    src = RAW / "bitcoin.csv"
    out = INTERIM / "bitcoin_aligned.csv"
    btc = read_bitcoin(src)
    btc.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(
        f"Span: {btc['date'].min().date()} → {btc['date'].max().date()} "
        f"rows: {len(btc)}\nColumns: {btc.columns.tolist()}"
    )

if __name__ == "__main__":
    main()
