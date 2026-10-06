"""
Load, merge, feature-engineer, and chronologically split the GEFCom2014 electricity-price
dataset (real substitute for the source paper's unreproducible synthetic price target — see
SOURCE_PAPER_AUDIT.md). Produces DATA_SPLIT_MANIFEST.csv (Sec. 13 requirement) and a clean
feature table used by every downstream benchmark/candidate model.
"""
import pandas as pd
import numpy as np

TASK15 = r"C:\Users\MMASSAOUDI\Desktop\Data\Load Data\GEFCom2014_Dataset\GEFCom2014 Data\GEFCom2014-P_V2\Price\Task 15\Task15_P.csv"
SOLUTION = r"C:\Users\MMASSAOUDI\Desktop\Data\Load Data\GEFCom2014_Dataset\GEFCom2014 Data\GEFCom2014-P_V2\Price\Solution to Task15\Solution to Task15_P.csv"

LAGS = [1, 2, 3, 24, 48, 168]     # hours: 1-3h short lag, daily, 2-day, weekly
ROLL_WINDOWS = [24, 168]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(TASK15)
    sol = pd.read_csv(SOLUTION)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m%d%Y %H:%M")
    sol["timestamp"] = pd.to_datetime(sol["timestamp"], format="%m%d%Y %H:%M")
    df = df.merge(sol, on=["ZONEID", "timestamp"], how="left", suffixes=("", "_sol"))
    df["Zonal Price"] = df["Zonal Price"].fillna(df["Zonal Price_sol"])
    df = df.drop(columns=["Zonal Price_sol", "ZONEID"]).sort_values("timestamp").reset_index(drop=True)
    df = df.rename(columns={
        "Forecasted Total Load": "system_load",
        "Forecasted Zonal Load": "zonal_load",
        "Zonal Price": "price",
    })
    assert df["price"].isna().sum() == 0, "unfilled price NaNs remain"
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    for lag in LAGS:
        df[f"load_lag{lag}"] = df["system_load"].shift(lag)
        df[f"zload_lag{lag}"] = df["zonal_load"].shift(lag)
        df[f"price_lag{lag}"] = df["price"].shift(lag)
    for w in ROLL_WINDOWS:
        df[f"load_roll_mean{w}"] = df["system_load"].shift(1).rolling(w).mean()
        df[f"price_roll_mean{w}"] = df["price"].shift(1).rolling(w).mean()
        df[f"price_roll_std{w}"] = df["price"].shift(1).rolling(w).std()

    df["load_ratio_zonal_total"] = df["zonal_load"] / df["system_load"].clip(lower=1)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = None  # set after first call to get_dataset()


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    return train, val, test


def get_dataset():
    global FEATURE_COLS
    raw = load_raw()
    feat = add_features(raw)
    exclude = {"timestamp", "price", "hour", "dow", "month"}
    FEATURE_COLS = [c for c in feat.columns if c not in exclude]
    train, val, test = chronological_split(feat)
    return feat, train, val, test, FEATURE_COLS


def save_manifest():
    feat, train, val, test, cols = get_dataset()
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rows.append({
            "split": name,
            "n_rows": len(part),
            "start_timestamp": part["timestamp"].min(),
            "end_timestamp": part["timestamp"].max(),
            "price_mean": part["price"].mean(),
            "price_std": part["price"].std(),
            "price_min": part["price"].min(),
            "price_max": part["price"].max(),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv("DATA_SPLIT_MANIFEST.csv", index=False)
    print(manifest.to_string(index=False))
    # duplicate check across splits (exact timestamp overlap)
    ts_train = set(train["timestamp"]); ts_val = set(val["timestamp"]); ts_test = set(test["timestamp"])
    assert not (ts_train & ts_val) and not (ts_val & ts_test) and not (ts_train & ts_test), "split leakage!"
    print("No cross-split timestamp overlap confirmed.")
    return manifest


if __name__ == "__main__":
    save_manifest()
