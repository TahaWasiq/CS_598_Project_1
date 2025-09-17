#!/usr/bin/env python3
"""
build-features.py — build model features/targets from bitcoin_aligned.csv
Reads : data/interim/bitcoin_aligned.csv
Writes: data/processed/btc_features_h1_full.csv, btc_features_h7_full.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    #keeps interface but uses a simple volatility proxy
    tr = close.diff().abs()
    return tr.rolling(n).mean()

def build_for_horizon(df: pd.DataFrame, h: int) -> pd.DataFrame:
    out = df.copy()

    #target: future close h days ahead
    out[f"y_btc_close_t+{h}"] = out["btc_close"].shift(-h)

    # basic price/return lags
    for k in range(1, 15):
        out[f"btc_close_lag{k}"] = out["btc_close"].shift(k)
        out[f"btc_ret1_lag{k}"] = out["btc_ret1"].shift(k)
        out[f"btc_volume_lag{k}"] = out["btc_volume"].shift(k)

    for k in (7, 14, 30):
        out[f"btc_roll_mean_close_{k}"] = out["btc_close"].rolling(k).mean()
        out[f"btc_roll_std_close_{k}"] = out["btc_close"].rolling(k).std()
        out[f"btc_roll_mean_ret_{k}"] = out["btc_ret1"].rolling(k).mean()
        out[f"btc_roll_vol_ret_{k}"] = out["btc_ret1"].rolling(k).std()
        out[f"btc_roll_mean_vol_{k}"] = out["btc_volume"].rolling(k).mean()
        out[f"btc_roll_std_vol_{k}"] = out["btc_volume"].rolling(k).std()

    out["btc_ema12"] = out["btc_close"].ewm(span=12, adjust=False).mean()
    out["btc_ema26"] = out["btc_close"].ewm(span=26, adjust=False).mean()
    out["btc_macd"] = out["btc_ema12"] - out["btc_ema26"]
    out["btc_rsi14"] = rsi(out["btc_close"], 14)
    out["btc_atr14"] = atr(out["btc_close"], out["btc_close"], out["btc_close"], 14)

    #drop rows that don’t have full history or target
    out = out.dropna().reset_index(drop=True)

    #keep date for tracing but as a normal column
    out = out[["date"] + [c for c in out.columns if c != "date"]]
    return out

def main():
    src = INTERIM / "bitcoin_aligned.csv"
    df = pd.read_csv(src, parse_dates=["date"]).sort_values("date")

    for h in (1, 7):
        full = build_for_horizon(df, h)
        out = PROCESSED / f"btc_features_h{h}_full.csv"
        full.to_csv(out, index=False)
        print(f"[h={h}] Saved: {out} rows: {len(full)}")

if __name__ == "__main__":
    main()
