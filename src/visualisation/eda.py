import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


def plot_price_and_returns(model_df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(model_df.index, model_df["close_cl=f"], linewidth=0.8, color="#1a1a2e")
    axes[0].set_title("WTI Crude Oil Futures — Daily Close Price")
    axes[0].set_ylabel("Price (USD)")

    events = {
        "2014-11-28": "OPEC price war",
        "2016-02-11": "Oil bottom ($26)",
        "2020-04-20": "Negative oil prices",
        "2022-03-08": "Russia-Ukraine spike",
    }
    for date_str, label in events.items():
        date = pd.Timestamp(date_str)
        nearest = model_df.index[model_df.index.get_indexer([date], method="nearest")[0]]
        price = model_df.loc[nearest, "close_cl=f"]
        axes[0].annotate(
            label, xy=(nearest, price), xytext=(0, 30), textcoords="offset points",
            fontsize=8, ha="center", color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    axes[1].set_ylim(-30, 30)
    axes[1].bar(
        model_df.index, model_df["wti_return"] * 100,
        width=1, color="steelblue", alpha=0.6, linewidth=0,
    )
    axes[1].set_ylabel("Daily Return (%)")
    axes[1].set_title("WTI Daily Returns")
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("outputs/price_and_returns.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_return_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    returns = df["wti_return"].dropna()

    axes[0].hist(returns * 100, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    x = np.linspace(returns.min() * 100, returns.max() * 100, 200)
    normal_pdf = stats.norm.pdf(x, loc=returns.mean() * 100, scale=returns.std() * 100)
    axes[0].plot(x, normal_pdf, "r-", linewidth=2, label="Normal fit")
    axes[0].set_title("Return Distribution vs Normal")
    axes[0].set_xlabel("Daily Return (%)")
    axes[0].legend()

    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot — Returns vs Normal")
    axes[1].get_lines()[0].set_markersize(2)

    yearly = df[["wti_return"]].copy()
    yearly["year"] = yearly.index.year
    yearly.boxplot(column="wti_return", by="year", ax=axes[2], rot=45, flierprops=dict(markersize=2))
    axes[2].set_title("Return Distribution by Year")
    axes[2].set_xlabel("")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig("outputs/return_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Mean:     {returns.mean()*100:.4f}%")
    print(f"Std Dev:  {returns.std()*100:.3f}%")
    print(f"Skewness: {returns.skew():.3f}")
    print(f"Kurtosis: {returns.kurtosis():.3f}")


def plot_autocorrelation(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    returns = df["wti_return"].dropna()

    plot_acf(returns, lags=40, ax=axes[0, 0], title="ACF — Daily Returns")
    plot_pacf(returns, lags=40, ax=axes[0, 1], title="PACF — Daily Returns", method="ywm")
    plot_acf(returns**2, lags=40, ax=axes[1, 0], title="ACF — Squared Returns")
    plot_acf(returns.abs(), lags=40, ax=axes[1, 1], title="ACF — Absolute Returns")

    plt.tight_layout()
    plt.savefig("outputs/autocorrelation.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_adf_tests(df):
    groups = {
        "Raw prices": [
            "close_cl=f", "close_bz=f", "close_ng=f", "close_^gspc",
            "usd_index", "cpi_raw", "industrial_prod", "inv",
        ],
        "Returns & diffs": [
            "wti_return", "brent_return", "natgas_return", "sp500_return",
            "usd_return", "ffr_diff", "y10_diff",
        ],
        "Rates": ["fed_funds_rate", "us_10y_yield", "t10y2y", "close_^vix"],
    }

    print("=" * 70)
    print("AUGMENTED DICKEY-FULLER STATIONARITY TESTS")
    print("=" * 70)

    for group_name, var_list in groups.items():
        print(f"\n--- {group_name} ---")
        for col in var_list:
            if col not in df.columns:
                continue
            result = adfuller(df[col].dropna(), maxlag=21, autolag="AIC")
            stat, pval = result[0], result[1]
            status = "STATIONARY" if pval < 0.05 else "NON-STATIONARY"
            print(f"  {col:25s}  ADF={stat:8.3f}  p={pval:.4f}  {status}")


def plot_correlation_matrix(df):
    cols = [c for c in [
        "wti_return", "brent_return", "natgas_return", "sp500_return", "usd_return",
        "ffr_diff", "y10_diff", "t10y2y_diff", "baa_diff",
        "inv_change_pct", "cpi_mom", "indprod_mom", "sentiment_mom",
        "brent_wti_spread", "credit_spread", "wti_range",
        "wti_roc5", "wti_roc21", "wti_vs_ma21", "wti_vs_ma63",
    ] if c in df.columns]

    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 8}, linewidths=0.5,
    )
    ax.set_title("Correlation Matrix — WTI Return vs Key Features")
    plt.tight_layout()
    plt.savefig("outputs/correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_rolling_correlations(df):
    vars_list = [
        ("inv", "Inventory"),
        ("usd_index", "USD Index"),
        ("close_^vix", "VIX"),
        ("close_^gspc", "S&P 500"),
        ("close_ng=f", "Natural Gas"),
        ("credit_spread", "Credit Spread"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, (col, label) in enumerate(vars_list):
        if col not in df.columns or i >= len(axes):
            continue
        roll_corr = df["close_cl=f"].rolling(252).corr(df[col])
        axes[i].plot(df.index, roll_corr, linewidth=0.8, color="steelblue")
        axes[i].axhline(0, color="black", linewidth=0.5)
        axes[i].fill_between(df.index, roll_corr, 0, alpha=0.2, color="steelblue")
        axes[i].set_title(f"Rolling 1Y Corr: WTI vs {label}", fontsize=10)
        axes[i].set_ylabel("Correlation")
        axes[i].set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig("outputs/rolling_correlations.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_volatility(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df["close_cl=f"], linewidth=0.7, color="#1a1a2e")
    axes[0].set_ylabel("WTI Price")
    axes[0].set_title("WTI Price, Realised Volatility, and VIX")

    if "wti_std21" in df.columns:
        ann_vol = df["wti_std21"] / df["close_cl=f"] * np.sqrt(252) * 100
        axes[1].plot(df.index, ann_vol, linewidth=0.7, color="#e74c3c")
        axes[1].set_ylabel("Annualised Vol (%)")
        axes[1].axhline(ann_vol.median(), color="grey", linestyle="--", linewidth=0.5)

    if "close_^vix" in df.columns:
        axes[2].plot(df.index, df["close_^vix"], linewidth=0.7, color="#8e44ad")
        axes[2].set_ylabel("VIX")
        axes[2].axhline(20, color="grey", linestyle="--", linewidth=0.5)
        axes[2].axhline(30, color="red", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("outputs/volatility.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_hp_filter(df):
    if not all(c in df.columns for c in ["hp_trend", "hp_cycle", "hp_cycle_pct"]):
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df["close_cl=f"], linewidth=0.7, color="grey", alpha=0.5, label="Actual")
    axes[0].plot(df.index, df["hp_trend"], linewidth=1.5, color="#e74c3c", label="HP Trend")
    axes[0].set_title("HP Filter — Trend Extraction")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()

    axes[1].plot(df.index, df["hp_cycle"], linewidth=0.7, color="steelblue")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("HP Cycle Component")
    axes[1].set_ylabel("Deviation (USD)")
    axes[1].fill_between(df.index, df["hp_cycle"], 0, alpha=0.2, color="steelblue")

    axes[2].plot(df.index, df["hp_cycle_pct"] * 100, linewidth=0.7, color="#8e44ad")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_title("HP Cycle as % of Trend")
    axes[2].set_ylabel("% Deviation")
    axes[2].fill_between(df.index, df["hp_cycle_pct"] * 100, 0, alpha=0.2, color="#8e44ad")

    plt.tight_layout()
    plt.savefig("outputs/hp_filter.png", dpi=150, bbox_inches="tight")
    plt.show()
