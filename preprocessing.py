import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import TRAIN_END, VAL_END, TARGET_COL, EXCLUDE_COLS, SEQ_LEN, BATCH_SIZE


def clean_and_build_model_df(df):
    df = df.dropna()

    price_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["close_", "high_", "low_", "open_", "volume_"]
    )]
    macro_cols = [c for c in [
        "usd_index", "fed_funds_rate", "us_10y_yield", "t10y2y",
        "us_2y_yield", "baa_yield", "cpi_raw", "ppi_raw",
        "industrial_prod", "umich_sentiment",
    ] if c in df.columns]
    eia_cols = [c for c in ["spot", "inv"] if c in df.columns]

    return_cols = [c for c in df.columns if "return" in c and "lag" not in c]
    diff_cols = [c for c in df.columns if "_diff" in c]
    change_cols = [c for c in df.columns if "_mom" in c or "inv_change" in c or "sentiment_mom" in c]
    spread_cols = [c for c in df.columns if "spread" in c and "lag" not in c]
    range_cols = [c for c in df.columns if "wti_range" in c]
    rolling_cols = [c for c in df.columns if any(
        x in c for x in ["_ma", "_std", "_roc", "_vs_ma", "_vol_ma", "_vol_ratio"]
    ) and "lag" not in c]
    decomp_cols = [c for c in df.columns if any(x in c for x in ["hp_", "ssa_"])]
    lag_cols = [c for c in df.columns if "lag" in c]
    cal_cols = [c for c in df.columns if c in [
        "day_of_week", "month", "quarter", "is_friday", "is_month_end", "is_eia_day",
    ]]

    keep_prices = [c for c in [
        "close_cl=f", "high_cl=f", "low_cl=f", "volume_cl=f",
        "close_bz=f", "close_ng=f", "close_^gspc", "close_^vix",
    ] if c in df.columns]

    feature_cols = (
        keep_prices
        + return_cols + diff_cols + change_cols
        + macro_cols + eia_cols
        + spread_cols + range_cols
        + rolling_cols + decomp_cols
        + lag_cols + cal_cols
    )

    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    model_df = df[feature_cols].copy()
    for col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    return model_df


def split_data(model_df):
    train_df = model_df[model_df.index <= TRAIN_END].copy()
    val_df = model_df[(model_df.index > TRAIN_END) & (model_df.index <= VAL_END)].copy()
    test_df = model_df[model_df.index > VAL_END].copy()
    return train_df, val_df, test_df


def get_feature_cols(model_df):
    return [c for c in model_df.columns if c not in EXCLUDE_COLS]


def scale_features(train_df, val_df, test_df, feature_cols):
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[feature_cols])
    val_X = scaler.transform(val_df[feature_cols])
    test_X = scaler.transform(test_df[feature_cols])

    train_y = train_df[TARGET_COL].values
    val_y = val_df[TARGET_COL].values
    test_y = test_df[TARGET_COL].values

    return scaler, train_X, val_X, test_X, train_y, val_y, test_y


class OilDataset(Dataset):
    def __init__(self, features, target, seq_len=SEQ_LEN):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.seq_len]
        y = self.target[idx + self.seq_len]
        return X, y


def create_dataloaders(train_X, val_X, test_X, train_y, val_y, test_y):
    train_ds = OilDataset(train_X, train_y, SEQ_LEN)
    val_ds = OilDataset(val_X, val_y, SEQ_LEN)
    test_ds = OilDataset(test_X, test_y, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
