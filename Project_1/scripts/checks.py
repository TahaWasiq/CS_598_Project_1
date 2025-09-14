import pandas as pd
from pathlib import Path

P = Path("data/processed")

def summarize(name, ycol):
    df = pd.read_csv(P / name, parse_dates=["date"])
    print(f"\n{name}")
    print("rows:", len(df), "cols:", len(df.columns))
    print("span:", df["date"].min().date(), "→", df["date"].max().date())
    na = df.isna().sum().sort_values(ascending=False)
    print("top NA:\n", na.head(8))
    print("target describe:\n", df[ycol].describe())

for h in [1, 7]:
    y = f"y_btc_close_t+{h}"
    for split in ["train","val","test"]:
        summarize(f"h{h}_{split}.csv", y)
