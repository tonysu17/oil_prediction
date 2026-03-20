import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

from config import (
    EIA_API_KEY, FRED_API_KEY, START_DATE, END_DATE, CUTOFF_DATE,
    YF_TICKERS, FRED_SERIES,
)


def get_eia_data(params, url, name):
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"EIA API request failed ({r.status_code}) for {name}")
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data).set_index("period")
    df.index = pd.to_datetime(df.index)
    df = df.resample("D").last().ffill()
    df = df[df.index < CUTOFF_DATE]["value"]
    return df.rename(name)


def fetch_eia():
    spot_params = {
        "api_key": EIA_API_KEY,
        "data[]": "value",
        "facets[product][]": "EPCWTI",
        "facets[series][]": "RWTC",
        "frequency": "weekly",
        "start": START_DATE,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    inv_params = {
        "api_key": EIA_API_KEY,
        "data[]": "value",
        "facets[product][]": "EPC0",
        "facets[process][]": "SAE",
        "frequency": "weekly",
        "start": START_DATE,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }
    spot_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    inv_url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"

    spot_df = get_eia_data(spot_params, spot_url, "spot")
    inv_df = get_eia_data(inv_params, inv_url, "inv")
    return pd.concat([spot_df, inv_df], axis=1)


def fetch_yfinance():
    frames = {}
    for ticker, name in YF_TICKERS.items():
        data = yf.download(tickers=ticker, start=START_DATE, end=END_DATE, interval="1d")
        df = pd.DataFrame(data)
        df = df[df.index < CUTOFF_DATE]
        frames[name] = df
    return pd.concat(frames.values(), axis=1)


def fetch_fred():
    fred = Fred(api_key=FRED_API_KEY)
    combined = {}
    for col_name, series_id in FRED_SERIES.items():
        series = fred.get_series(
            series_id, observation_start=START_DATE, observation_end=END_DATE
        )
        combined[col_name] = series.resample("D").last().ffill()
    df = pd.DataFrame(combined)
    df = df[df.index < CUTOFF_DATE]
    return df.ffill()


def fetch_all():
    eia_df = fetch_eia()
    yf_df = fetch_yfinance()
    fred_df = fetch_fred()

    df = pd.concat([eia_df, fred_df, yf_df], axis=1)
    df = df.ffill().dropna()
    df.columns = [
        f"{col[0]}_{col[1]}".lower() if isinstance(col, tuple) else col
        for col in df.columns
    ]
    return df
