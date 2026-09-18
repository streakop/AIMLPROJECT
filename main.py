"""
Stock Direction Classifier — a simple, complete AI/ML project.

Goal: predict whether a stock's closing price will go UP or DOWN
the next trading day, using only past price/volume-derived features.

Pipeline:
  1. Generate a realistic synthetic OHLCV price series
     (no internet/data feed required — fully self-contained).
  2. Engineer features: returns, moving averages, volatility, RSI, volume trend.
  3. Label: 1 if next day's close > today's close, else 0.
  4. Train/test split (time-based, not random — avoids lookahead leakage).
  5. Train a RandomForest classifier.
  6. Evaluate: accuracy, precision/recall, confusion matrix.
  7. Plot price series + predicted vs actual direction, save as PNG.

Run:
    python main.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. Generate synthetic OHLCV data
# ---------------------------------------------------------------------------
def generate_synthetic_prices(n_days: int = 1000, start_price: float = 500.0) -> pd.DataFrame:
    """
    Simulates a stock price series using a mean-reverting drift + random walk,
    with volume that reacts to volatility (more volume on bigger moves) —
    mimicking real market microstructure loosely.
    """
    dates = pd.bdate_range(start="2021-01-01", periods=n_days)

    # daily returns: small drift + noise + occasional volatility clusters
    base_vol = 0.012
    vol_regime = base_vol * (1 + 0.5 * np.sin(np.linspace(0, 15, n_days)))
    drift = 0.0003
    daily_returns = np.random.normal(loc=drift, scale=vol_regime, size=n_days)

    close = start_price * np.cumprod(1 + daily_returns)

    # derive open/high/low around close with small intraday noise
    intraday_noise = np.random.normal(0, base_vol / 2, size=n_days)
    open_ = close * (1 - intraday_noise)
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, base_vol / 2, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, base_vol / 2, n_days)))

    volume = (1_000_000 * (1 + 8 * np.abs(daily_returns) / base_vol)).astype(int)
    volume += np.random.randint(-50_000, 50_000, n_days)
    volume = np.clip(volume, 100_000, None)

    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }).set_index("date")

    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    df["volatility_10d"] = df["return_1d"].rolling(10).std()

    df["rsi_14"] = compute_rsi(df["close"], 14)

    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(10).mean()

    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]

    # Target: did price go UP the next trading day?
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.dropna()
    return df


FEATURE_COLS = [
    "return_1d", "return_5d", "ma_ratio", "volatility_10d",
    "rsi_14", "volume_change", "volume_ma_ratio", "high_low_spread",
]


# ---------------------------------------------------------------------------
# 3. Train / evaluate
# ---------------------------------------------------------------------------
def main():
    print("Generating synthetic price data...")
    raw = generate_synthetic_prices(n_days=1000)

    print("Engineering features...")
    data = build_features(raw)

    X = data[FEATURE_COLS]
    y = data["target"]

    # Time-based split: train on first 80%, test on last 20% (no shuffling —
    # this avoids leaking future information into training, unlike a random split).
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    baseline = max(y_test.mean(), 1 - y_test.mean())  # always-predict-majority baseline

    print("\n=== Results ===")
    print(f"Model accuracy:     {acc:.3f}")
    print(f"Majority baseline:  {baseline:.3f}  (accuracy if you always guessed the more common class)")
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["DOWN", "UP"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds))

    print("\nFeature importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances.to_string())

    # -----------------------------------------------------------------
    # 4. Plot
    # -----------------------------------------------------------------
    test_dates = data.index[split_idx:]
    test_close = raw.loc[test_dates, "close"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(test_dates, test_close, color="steelblue", label="Close price (test period)")
    axes[0].set_title("Synthetic Stock Price — Test Period")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    correct = (preds == y_test.values)
    axes[1].scatter(test_dates, preds, c=np.where(correct, "green", "red"), s=15,
                     label="Predicted direction (green=correct, red=wrong)")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["DOWN", "UP"])
    axes[1].set_title(f"Predicted vs Actual Direction (accuracy={acc:.1%})")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = "results.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
