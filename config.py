import os

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
EIA_API_KEY = os.environ.get("EIA_API_KEY", "")

START_DATE = "2010-01-01"
END_DATE = "2026-03-06"
CUTOFF_DATE = "2026-03-01"

TRAIN_END = "2022-12-31"
VAL_END = "2024-06-30"

SEQ_LEN = 60
BATCH_SIZE = 64
TARGET_COL = "wti_return"

YF_TICKERS = {
    "CL=F": "wti",
    "BZ=F": "brent",
    "NG=F": "natgas",
    "^GSPC": "sp500",
    "^VIX": "vix",
}

FRED_SERIES = {
    "usd_index": "DTWEXBGS",
    "fed_funds_rate": "DFF",
    "us_10y_yield": "DGS10",
    "t10y2y": "T10Y2Y",
    "us_2y_yield": "DGS2",
    "baa_yield": "DBAA",
    "cpi_raw": "CPIAUCSL",
    "ppi_raw": "PPIACO",
    "industrial_prod": "INDPRO",
    "umich_sentiment": "UMCSENT",
}

EXCLUDE_COLS = {
    TARGET_COL,
    "close_cl=f",
    "ssa_residual",
}
