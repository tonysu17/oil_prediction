import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(pred_returns, actual_returns, prev_prices, actual_prices, model_name):
    pred_prices = prev_prices * (1 + pred_returns)

    ret_mae = mean_absolute_error(actual_returns, pred_returns)
    ret_rmse = np.sqrt(mean_squared_error(actual_returns, pred_returns))
    price_mae = mean_absolute_error(actual_prices, pred_prices)
    price_rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))

    dir_correct = np.sign(pred_returns) == np.sign(actual_returns)
    dir_acc = dir_correct.mean()

    naive_mae = mean_absolute_error(actual_prices, prev_prices)

    results = {
        "model": model_name,
        "return_mae": ret_mae,
        "return_rmse": ret_rmse,
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "directional_accuracy": dir_acc,
        "naive_price_mae": naive_mae,
        "beats_naive": price_mae < naive_mae,
    }

    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Return MAE:           {ret_mae*100:.4f}%")
    print(f"  Return RMSE:          {ret_rmse*100:.4f}%")
    print(f"  Price MAE:            ${price_mae:.2f}")
    print(f"  Price RMSE:           ${price_rmse:.2f}")
    print(f"  Directional Accuracy: {dir_acc*100:.1f}%")
    print(f"  Naive Price MAE:      ${naive_mae:.2f}")
    print(f"  Beats naive:          {'YES' if price_mae < naive_mae else 'NO'}")

    return results, pred_prices


def plot_all_predictions(
    test_dates_tab, test_prices_tab, test_dates_seq, test_prices_seq, all_pred_prices
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    colors = {
        "Prophet": "#e74c3c",
        "XGBoost": "#27ae60",
        "LSTM": "#3498db",
        "Transformer": "#8e44ad",
    }

    ax1 = axes[0]
    ax1.plot(test_dates_tab, test_prices_tab, "k-", linewidth=1.2, label="Actual", alpha=0.9)

    for name in ["Prophet", "XGBoost"]:
        if name in all_pred_prices:
            ax1.plot(
                test_dates_tab, all_pred_prices[name],
                linewidth=0.8, alpha=0.7, color=colors[name], label=name,
            )
    for name in ["LSTM", "Transformer"]:
        if name in all_pred_prices:
            ax1.plot(
                test_dates_seq, all_pred_prices[name],
                linewidth=0.8, alpha=0.7, color=colors[name], label=name,
            )

    ax1.set_ylabel("WTI Price (USD)")
    ax1.set_title("Test Set — Predicted vs Actual WTI Price")
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    for name in ["Prophet", "XGBoost"]:
        if name in all_pred_prices:
            err = test_prices_tab - all_pred_prices[name]
            ax2.plot(test_dates_tab, err, linewidth=0.7, alpha=0.6, color=colors[name], label=name)
    for name in ["LSTM", "Transformer"]:
        if name in all_pred_prices:
            err = test_prices_seq - all_pred_prices[name]
            ax2.plot(test_dates_seq, err, linewidth=0.7, alpha=0.6, color=colors[name], label=name)

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Prediction Error (USD)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/prediction_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_bar_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("price_mae", "Price MAE ($)", True),
        ("price_rmse", "Price RMSE ($)", True),
        ("directional_accuracy", "Directional Accuracy", False),
    ]

    for ax, (metric, label, lower_better) in zip(axes, metrics):
        values = results_df[metric]
        best = values.min() if lower_better else values.max()
        colors_bar = ["#27ae60" if v == best else "steelblue" for v in values]

        ax.bar(values.index, values.values, color=colors_bar, alpha=0.8, edgecolor="white")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=15)

        for i, (model, v) in enumerate(values.items()):
            fmt = f"{v:.2f}" if "accuracy" not in metric else f"{v*100:.1f}%"
            ax.text(
                i, v + (values.max() - values.min()) * 0.02, fmt,
                ha="center", fontsize=9, fontweight="bold" if v == best else "normal",
            )

    plt.suptitle("Model Comparison — Test Set", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
