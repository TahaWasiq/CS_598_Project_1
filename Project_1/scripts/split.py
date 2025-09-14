import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def split_scale_save(h, train_end="2019-12-31", val_start="2020-01-01", val_end="2020-12-31", test_start="2021-01-01"):
    df = pd.read_csv(PROCESSED / f"btc_features_h{h}_full.csv", parse_dates=["date"]).set_index("date").sort_index()
    ycol = f"y_btc_close_t+{h}"
    cols = [c for c in df.columns if c != ycol]
    train = df.loc[:train_end].copy()
    val   = df.loc[val_start:val_end].copy()
    test  = df.loc[test_start:].copy()
    train = train.dropna()
    val   = val.dropna()
    test  = test.dropna()
    Xtr, ytr = train[cols], train[ycol]
    Xva, yva = val[cols],   val[ycol]
    Xte, yte = test[cols],  test[ycol]
    means = Xtr.mean()
    stds  = Xtr.std(ddof=0).replace(0, 1.0)
    Xtr = (Xtr - means) / stds
    Xva = (Xva - means) / stds
    Xte = (Xte - means) / stds
    def pack(X, y):
        out = X.copy()
        out.insert(0, "date", X.index)
        out[ycol] = y.values
        return out.reset_index(drop=True)
    pack(Xtr, ytr).to_csv(PROCESSED / f"h{h}_train.csv", index=False)
    pack(Xva, yva).to_csv(PROCESSED / f"h{h}_val.csv", index=False)
    pack(Xte, yte).to_csv(PROCESSED / f"h{h}_test.csv", index=False)
    print(f"[h={h}] train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

for h in [1, 7]:
    split_scale_save(h)
