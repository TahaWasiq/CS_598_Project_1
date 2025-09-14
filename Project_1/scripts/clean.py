import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

def read_coin(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        for c in df.columns:
            if "date" in c:
                df = df.rename(columns={c: "date"})
                break
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for k in ["open","high","low","close","volume","marketcap"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "marketcap" in df.columns and "mktcap" not in df.columns:
        df["mktcap"] = df["marketcap"]
        df = df.drop(columns=["marketcap"])
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    keep = ["date","open","high","low","close","volume","mktcap"]
    return df[[c for c in keep if c in df.columns]]

def add_returns(df):
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    ql, qh = df["ret1"].quantile([0.005, 0.995])
    df["ret1"] = df["ret1"].clip(ql, qh)
    return df

def ffill_limit(df, cols, limit=2):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].ffill(limit=limit)
    return df

btc = read_coin(RAW / "coin_Bitcoin.csv").set_index("date")
eth = read_coin(RAW / "coin_Ethereum.csv").set_index("date")
ltc = read_coin(RAW / "coin_Litecoin.csv").set_index("date")

btc = add_returns(btc).asfreq("D")
eth = add_returns(eth).asfreq("D")
ltc = add_returns(ltc).asfreq("D")

btc = ffill_limit(btc, ["open","high","low","close","volume","mktcap","ret1"], 2)
eth = ffill_limit(eth, ["close","volume","mktcap","ret1"], 2)
ltc = ffill_limit(ltc, ["close","volume","mktcap","ret1"], 2)

aligned = pd.DataFrame(index=btc.index.copy())
for col in ["open","high","low","close","volume","mktcap","ret1"]:
    if col in btc.columns:
        aligned[f"btc_{col}"] = btc[col]
for p, df in [("eth", eth), ("ltc", ltc)]:
    for col in ["close","ret1","volume","mktcap"]:
        if col in df.columns:
            aligned[f"{p}_{col}"] = df[col]

aligned = aligned[~aligned["btc_close"].isna()]
aligned = aligned.reset_index().rename(columns={"index":"date"})
aligned.to_csv(INTERIM / "aligned_btc_eth_ltc.csv", index=False)

print("Saved:", INTERIM / "aligned_btc_eth_ltc.csv")
print("Span:", aligned["date"].min().date(), "→", aligned["date"].max().date(), "rows:", len(aligned))
print("Columns:", list(aligned.columns))