import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def rsi(close, n):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def atr(high, low, close, n):
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

df = pd.read_csv(INTERIM / "aligned_btc_eth_ltc.csv", parse_dates=["date"]).set_index("date").sort_index()

roll_windows = [7, 14, 30]
lag_max = 14
new_cols = {}

for k in roll_windows:
    new_cols[f"btc_roll_mean_close_{k}"] = df["btc_close"].rolling(k).mean()
    new_cols[f"btc_roll_std_close_{k}"]  = df["btc_close"].rolling(k).std()
    new_cols[f"btc_roll_mean_ret_{k}"]   = df["btc_ret1"].rolling(k).mean()
    new_cols[f"btc_roll_vol_ret_{k}"]    = df["btc_ret1"].rolling(k).std()
    new_cols[f"btc_roll_mean_vol_{k}"]   = df["btc_volume"].rolling(k).mean()
    new_cols[f"btc_roll_std_vol_{k}"]    = df["btc_volume"].rolling(k).std()

ema12 = df["btc_close"].ewm(span=12, adjust=False).mean()
ema26 = df["btc_close"].ewm(span=26, adjust=False).mean()
new_cols["btc_ema12"] = ema12
new_cols["btc_ema26"] = ema26
new_cols["btc_macd"]  = ema12 - ema26
new_cols["btc_rsi14"] = rsi(df["btc_close"], 14)
new_cols["btc_atr14"] = atr(df["btc_high"], df["btc_low"], df["btc_close"], 14)

assets = ["btc", "eth", "ltc"]
for asset in assets:
    for k in range(1, lag_max + 1):
        new_cols[f"{asset}_ret1_lag{k}"] = df[f"{asset}_ret1"].shift(k)
    for k in roll_windows:
        new_cols[f"{asset}_roll_mean_ret_{k}"] = df[f"{asset}_ret1"].rolling(k).mean()
        new_cols[f"{asset}_roll_vol_ret_{k}"]  = df[f"{asset}_ret1"].rolling(k).std()
    for k in range(1, lag_max + 1):
        if f"{asset}_volume" in df:
            new_cols[f"{asset}_volume_lag{k}"] = df[f"{asset}_volume"].shift(k)

df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

for h in [1, 7]:
    out = df.copy()
    out[f"y_btc_close_t+{h}"] = out["btc_close"].shift(-h)
    out = out.dropna()
    out.reset_index().to_csv(PROCESSED / f"btc_features_h{h}_full.csv", index=False)
    print(f"[h={h}] Saved:", PROCESSED / f"btc_features_h{h}_full.csv", "rows:", len(out))