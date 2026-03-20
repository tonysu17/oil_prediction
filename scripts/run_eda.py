import sys
sys.path.insert(0, ".")

from src.data.fetch import fetch_all
from src.data.features import engineer_features
from src.data.preprocessing import clean_and_build_model_df
from src.visualization.eda import (
    plot_price_and_returns,
    plot_return_distribution,
    plot_autocorrelation,
    run_adf_tests,
    plot_correlation_matrix,
    plot_rolling_correlations,
    plot_volatility,
    plot_hp_filter,
)


def main():
    print("Fetching data...")
    raw_df = fetch_all()

    print("Engineering features...")
    df = engineer_features(raw_df)

    print("Cleaning...")
    model_df = clean_and_build_model_df(df)
    df_clean = df.loc[model_df.index]

    print(f"Shape: {model_df.shape}")
    print(f"Date range: {model_df.index.min()} -> {model_df.index.max()}\n")

    plot_price_and_returns(model_df)
    plot_return_distribution(df_clean)
    plot_autocorrelation(df_clean)
    run_adf_tests(df_clean)
    plot_correlation_matrix(df_clean)
    plot_rolling_correlations(df_clean)
    plot_volatility(df_clean)
    plot_hp_filter(df_clean)


if __name__ == "__main__":
    main()
