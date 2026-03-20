import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

from config import TARGET_COL, SEQ_LEN
from src.data.fetch import fetch_all
from src.data.features import engineer_features
from src.data.preprocessing import (
    clean_and_build_model_df,
    split_data,
    get_feature_cols,
    scale_features,
    create_dataloaders,
)
from src.models.lstm import OilLSTM
from src.models.transformer import OilTransformer
from src.models.train import train_sequence_model
from src.models.evaluate import (
    evaluate_predictions,
    plot_all_predictions,
    plot_bar_comparison,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Fetching data...")
    raw_df = fetch_all()

    print("Engineering features...")
    df = engineer_features(raw_df)

    print("Building model dataframe...")
    model_df = clean_and_build_model_df(df)

    print("Splitting data...")
    train_df, val_df, test_df = split_data(model_df)
    feature_cols = get_feature_cols(model_df)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"  Features: {len(feature_cols)}\n")

    print("Scaling features...")
    scaler, train_X, val_X, test_X, train_y, val_y, test_y = scale_features(
        train_df, val_df, test_df, feature_cols
    )

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_X, val_X, test_X, train_y, val_y, test_y
    )

    test_prices_tab = test_df["close_cl=f"].values
    test_prev_tab = np.roll(test_df["close_cl=f"].values, 1)
    test_prev_tab[0] = train_df["close_cl=f"].values[-1]
    test_returns_tab = test_df[TARGET_COL].values
    test_dates_tab = test_df.index

    test_prices_seq = test_df["close_cl=f"].values[SEQ_LEN:]
    test_prev_seq = test_df["close_cl=f"].values[SEQ_LEN - 1 : -1]
    test_returns_seq = test_df[TARGET_COL].values[SEQ_LEN:]
    test_dates_seq = test_df.index[SEQ_LEN:]

    all_results = []
    all_pred_prices = {}

    # ── Prophet ──
    print("\n--- Prophet ---")
    prophet_train = pd.DataFrame({"ds": train_df.index, "y": train_df[TARGET_COL].values})
    prophet_model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    prophet_model.fit(prophet_train)
    prophet_forecast = prophet_model.predict(pd.DataFrame({"ds": test_df.index}))
    prophet_pred_returns = prophet_forecast["yhat"].values

    prophet_results, prophet_pred_prices = evaluate_predictions(
        prophet_pred_returns, test_returns_tab, test_prev_tab, test_prices_tab, "Prophet"
    )
    all_results.append(prophet_results)
    all_pred_prices["Prophet"] = prophet_pred_prices

    # ── XGBoost ──
    print("\n--- XGBoost ---")
    dtrain = xgb.DMatrix(train_df[feature_cols].values, label=train_y, feature_names=feature_cols)
    dval = xgb.DMatrix(val_df[feature_cols].values, label=val_y, feature_names=feature_cols)
    dtest = xgb.DMatrix(test_df[feature_cols].values, feature_names=feature_cols)

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
        "verbosity": 0,
    }
    xgb_model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    xgb_pred_returns = xgb_model.predict(dtest)

    xgb_results, xgb_pred_prices = evaluate_predictions(
        xgb_pred_returns, test_returns_tab, test_prev_tab, test_prices_tab, "XGBoost"
    )
    all_results.append(xgb_results)
    all_pred_prices["XGBoost"] = xgb_pred_prices

    # ── LSTM ──
    print("\n--- LSTM ---")
    input_dim = train_X.shape[1]
    lstm_model = OilLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    lstm_model, lstm_history = train_sequence_model(
        lstm_model, train_loader, val_loader, device,
        lr=1e-3, weight_decay=1e-5, epochs=100, patience=15,
    )

    lstm_model.eval()
    lstm_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            pred = lstm_model(X_batch.to(device))
            lstm_preds.extend(pred.cpu().numpy())
    lstm_pred_returns = np.array(lstm_preds)

    lstm_results, lstm_pred_prices = evaluate_predictions(
        lstm_pred_returns, test_returns_seq, test_prev_seq, test_prices_seq, "LSTM"
    )
    all_results.append(lstm_results)
    all_pred_prices["LSTM"] = lstm_pred_prices

    # ── Transformer ──
    print("\n--- Transformer ---")
    transformer_model = OilTransformer(
        input_dim=input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.3
    ).to(device)
    transformer_model, transformer_history = train_sequence_model(
        transformer_model, train_loader, val_loader, device,
        lr=5e-4, weight_decay=1e-5, epochs=50, patience=15,
    )

    transformer_model.eval()
    transformer_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            pred = transformer_model(X_batch.to(device))
            transformer_preds.extend(pred.cpu().numpy())
    transformer_pred_returns = np.array(transformer_preds)

    transformer_results, transformer_pred_prices = evaluate_predictions(
        transformer_pred_returns, test_returns_seq, test_prev_seq, test_prices_seq, "Transformer"
    )
    all_results.append(transformer_results)
    all_pred_prices["Transformer"] = transformer_pred_prices

    # ── Summary ──
    results_df = pd.DataFrame(all_results).set_index("model")
    print("\n" + "=" * 75)
    print("MODEL COMPARISON — TEST SET")
    print("=" * 75)
    for _, row in results_df.iterrows():
        print(
            f"\n{row.name}:  "
            f"Price MAE: ${row['price_mae']:.2f}  |  "
            f"Price RMSE: ${row['price_rmse']:.2f}  |  "
            f"Direction: {row['directional_accuracy']*100:.1f}%  |  "
            f"Beats naive: {'YES' if row['beats_naive'] else 'NO'}"
        )

    plot_all_predictions(
        test_dates_tab, test_prices_tab, test_dates_seq, test_prices_seq, all_pred_prices
    )
    plot_bar_comparison(results_df)


if __name__ == "__main__":
    main()
