# WTI Crude Oil Price Prediction

Forecasting next-day WTI crude oil returns using multi-source financial data and four competing models: **Prophet**, **XGBoost**, **LSTM**, and a **Transformer encoder**.

## Project Overview

This project builds an end-to-end pipeline that ingests macroeconomic, energy-market, and financial data from three APIs (EIA, FRED, Yahoo Finance), engineers 80+ features, and benchmarks four models on their ability to predict daily WTI crude oil returns. Predicted returns are converted back to price levels for evaluation.

### Key Features

- **Multi-source data ingestion** from EIA (spot prices, inventories), FRED (rates, macro indicators), and Yahoo Finance (futures, equities, VIX)
- **Extensive feature engineering**: returns, rate first-differences, rolling statistics, Hodrick-Prescott decomposition, lagged features, spreads, and volume ratios
- **Rigorous temporal train/validation/test split** (no lookahead bias)
- **Four model architectures** compared on identical test data
- **Evaluation metrics**: return MAE/RMSE, price MAE/RMSE, directional accuracy, and naive-baseline comparison

## Repository Structure

```
oil-price-prediction/
├── config.py                          # API keys (env vars), dates, hyperparameters
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── fetch.py                   # EIA, Yahoo Finance, FRED data fetchers
│   │   ├── features.py                # Feature engineering pipeline
│   │   └── preprocessing.py           # Cleaning, train/val/test split, scaling, PyTorch Dataset
│   │
│   ├── models/
│   │   ├── lstm.py                    # LSTM architecture
│   │   ├── transformer.py            # Transformer encoder architecture
│   │   ├── train.py                   # Training loop with early stopping
│   │   └── evaluate.py               # Metrics and prediction visualisation
│   │
│   └── visualization/
│       └── eda.py                     # Exploratory data analysis plots
│
├── scripts/
│   ├── run_pipeline.py                # End-to-end: fetch → engineer → train → evaluate
│   └── run_eda.py                     # Run all EDA visualisations
│
└── outputs/                           # Saved plots and figures
```

## Setup

```bash
git clone https://github.com/<your-username>/oil-price-prediction.git
cd oil-price-prediction
pip install -r requirements.txt
```

Set your API keys as environment variables:

```bash
export FRED_API_KEY="your_fred_key"
export EIA_API_KEY="your_eia_key"
```

Free API keys can be obtained from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) and [EIA](https://www.eia.gov/opendata/register.php).

## Usage

**Run the full training pipeline:**

```bash
python scripts/run_pipeline.py
```

**Run exploratory data analysis only:**

```bash
python scripts/run_eda.py
```

## Models

| Model | Type | Input | Description |
|-------|------|-------|-------------|
| Prophet | Additive decomposition | Univariate returns | Baseline — captures weekly/yearly seasonality |
| XGBoost | Gradient-boosted trees | Flat tabular features | Interpretable via SHAP; no sequence modelling |
| LSTM | Recurrent neural network | 60-day sliding windows | Captures temporal dependencies and volatility clustering |
| Transformer | Self-attention encoder | 60-day sliding windows | Learns long-range dependencies via multi-head attention |

## Data Sources

- **EIA API**: WTI spot prices, US crude oil inventories (weekly)
- **FRED API**: Fed funds rate, Treasury yields, USD index, CPI, PPI, industrial production, consumer sentiment, credit spreads
- **Yahoo Finance**: WTI/Brent/natural gas futures, S&P 500, VIX (daily OHLCV)

## Methodology

1. **Data acquisition**: Fetch from three APIs, resample to daily, forward-fill lower-frequency series
2. **Feature engineering**: Compute returns, rate diffs, rolling means/stds/ROC, HP filter decomposition, lagged features, cross-asset spreads
3. **Temporal split**: Train (2010–2022), Validation (2023–mid 2024), Test (mid 2024–2026)
4. **Scaling**: StandardScaler fit on training set only
5. **Training**: Each model trained with early stopping on validation loss
6. **Evaluation**: Return and price-level MAE/RMSE, directional accuracy, comparison against naive (random walk) baseline
