import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter


def add_returns(df):
    return_cols = {
        "close_cl=f": "wti_return",
        "close_bz=f": "brent_return",
        "close_ng=f": "natgas_return",
        "close_^gspc": "sp500_return",
        "usd_index": "usd_return",
    }
    for raw_col, ret_col in return_cols.items():
        df[ret_col] = df[raw_col].pct_change()
    return df


def add_rate_diffs(df):
    diff_cols = {
        "fed_funds_rate": "ffr_diff",
        "us_10y_yield": "y10_diff",
        "t10y2y": "t10y2y_diff",
        "baa_yield": "baa_diff",
    }
    for raw_col, diff_col in diff_cols.items():
        if raw_col in df.columns:
            df[diff_col] = df[raw_col].diff()
    return df


def add_inventory_and_macro(df):
    df["inv"] = df["inv"].astype("float")
    df["inv_change_5d"] = df["inv"].diff(5)
    df["inv_change_pct"] = df["inv"].pct_change(5)

    df["cpi_mom"] = df["cpi_raw"].pct_change(21)
    df["indprod_mom"] = df["industrial_prod"].pct_change(21)
    df["sentiment_mom"] = df["umich_sentiment"].pct_change(21)

    for prefix in ["wti", "brent", "natgas", "sp500"]:
        vol_col = f"{prefix}_volume"
        if vol_col in df.columns:
            df[f"{prefix}_vol_ratio"] = df[vol_col] / df[vol_col].rolling(21).mean()
    return df


def add_spreads(df):
    df["brent_wti_spread"] = df["close_bz=f"] - df["close_cl=f"]
    df["credit_spread"] = df["baa_yield"] - df["us_10y_yield"]
    df["wti_range"] = (df["high_cl=f"] - df["low_cl=f"]) / df["close_cl=f"]
    return df


def add_rolling_stats(df):
    for w in [5, 21, 63]:
        df[f"wti_ma{w}"] = df["close_cl=f"].rolling(w).mean()
        df[f"wti_std{w}"] = df["close_cl=f"].rolling(w).std()
        df[f"wti_roc{w}"] = df["close_cl=f"].pct_change(w)

    df["wti_vs_ma5"] = df["close_cl=f"] / df["wti_ma5"] - 1
    df["wti_vs_ma21"] = df["close_cl=f"] / df["wti_ma21"] - 1
    df["wti_vs_ma63"] = df["close_cl=f"] / df["wti_ma63"] - 1

    df["vix_ma5"] = df["close_^vix"].rolling(5).mean()
    df["vix_ma21"] = df["close_^vix"].rolling(21).mean()

    df["usd_ma21"] = df["usd_index"].rolling(21).mean()
    df["usd_roc21"] = df["usd_index"].pct_change(21)

    df["wti_vol_ma5"] = df["volume_cl=f"].rolling(5).mean()
    df["wti_vol_ma21"] = df["volume_cl=f"].rolling(21).mean()
    return df


def add_hp_filter(df):
    cycle, trend = hpfilter(df["close_cl=f"].dropna(), lamb=129600)
    df["hp_trend"] = trend
    df["hp_cycle"] = cycle
    df["hp_cycle_pct"] = cycle / trend
    return df


def add_lags(df):
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"wti_close_lag{lag}"] = df["close_cl=f"].shift(lag)
        df[f"wti_return_lag{lag}"] = df["wti_return"].shift(lag)

    for lag in [1, 2, 3, 5]:
        df[f"vix_close_lag{lag}"] = df["close_^vix"].shift(lag)

    for lag in [5, 10]:
        df[f"inv_lag{lag}"] = df["inv"].shift(lag)

    df["usd_index_lag1"] = df["usd_index"].shift(1)
    df["usd_index_lag5"] = df["usd_index"].shift(5)
    df["credit_spread_lag1"] = df["credit_spread"].shift(1)
    df["credit_spread_lag5"] = df["credit_spread"].shift(5)
    return df


def engineer_features(df):
    df = add_returns(df)
    df = add_rate_diffs(df)
    df = add_inventory_and_macro(df)
    df = add_spreads(df)
    df = add_rolling_stats(df)
    df = add_hp_filter(df)
    df = add_lags(df)
    return df
