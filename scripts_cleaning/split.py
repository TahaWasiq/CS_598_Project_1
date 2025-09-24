"""
Purpose: Chronological 70/15/15 split + z-score scaling using TRAIN-only stats.
Reads: data/processed/btc_features_h1_full.csv, btc_features_h7_full.csv
Writes: data/processed/h1_{train,val,test}.csv and h7_{train,val,test}.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED = Path("../data/processed")

def autosplit_by_date(df: pd.DataFrame):
    """
    Chronological ~70/15/15 split using date quantiles.
    Ensures each split is non-empty (nudges boundaries if needed).
    """
    df = df.sort_values("date").reset_index(drop=True)
    q70 = df["date"].quantile(0.70)
    q85 = df["date"].quantile(0.85)

    train = df[df["date"] <= q70].copy()
    val   = df[(df["date"] > q70) & (df["date"] <= q85)].copy()
    test  = df[df["date"] > q85].copy()

    #if any split is empty, degrade to simple contiguous splits
    n = len(df)
    if min(len(train), len(val), len(test)) == 0:
        n_train = max(int(n * 0.70), 1)
        n_val   = max(int(n * 0.15), 1)
        train = df.iloc[:n_train].copy()
        val   = df.iloc[n_train:n_train + n_val].copy()
        test  = df.iloc[n_train + n_val:].copy()
        #if test somehow ends empty, move 1 row from val
        if len(test) == 0 and len(val) > 1:
            test = val.tail(1).copy()
            val  = val.iloc[:-1].copy()

    return train, val, test

def scale_with_train_stats(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target_col: str):
    """
    Z-score numerical features using TRAIN-only mean/std.
    Leaves 'date' and target columns unscaled.
    """
    keep_plain = ["date", target_col]
    feat_cols = [c for c in train.columns if c not in keep_plain]

    #compute stats on train
    mu = train[feat_cols].mean()
    sd = train[feat_cols].std().replace(0, 1.0)  # avoid divide-by-zero

    def zscore(frame):
        out = frame.copy()
        out[feat_cols] = (out[feat_cols] - mu) / sd
        cols = ["date"] + [c for c in out.columns if c != "date"]
        return out[cols]

    return zscore(train), zscore(val), zscore(test)

def run_for_h(h: int):
    fname = PROCESSED / f"btc_features_h{h}_full.csv"
    df = pd.read_csv(fname, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    target_col = f"y_btc_close_t+{h}"

    #drop rows with any NA (typical after rolling windows)
    df = df.dropna().reset_index(drop=True)

    #split
    train, val, test = autosplit_by_date(df)

    #train-only stats
    train_z, val_z, test_z = scale_with_train_stats(train, val, test, target_col)

    #save
    out_train = PROCESSED / f"h{h}_train.csv"
    out_val   = PROCESSED / f"h{h}_val.csv"
    out_test  = PROCESSED / f"h{h}_test.csv"
    train_z.to_csv(out_train, index=False)
    val_z.to_csv(out_val, index=False)
    test_z.to_csv(out_test, index=False)

    print(f"[h={h}] train={len(train_z)} val={len(val_z)} test={len(test_z)} "
          f"span: {df['date'].min().date()} → {df['date'].max().date()}")

if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for h in (1, 7):
        run_for_h(h)
