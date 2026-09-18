# Stock Direction Classifier

A simple, self-contained AI/ML project: predict whether a stock's closing
price will go **UP** or **DOWN** the next trading day, using only
price/volume-derived features.

## Why this project (and why it "underperforms")

Real next-day price direction is close to a random walk. This project
trains an honest baseline model and shows that clearly: the model
scored **48.7% accuracy** vs a **53.8% majority-class baseline** — i.e.
it did *worse* than just always guessing "up" (or whichever direction
was more common in the test window). That's not a bug — it's the
correct, expected result for this feature set, and it's a good
first lesson in why quant signal generation is hard: technical
features alone rarely beat a coin flip on raw next-day direction.

## Pipeline

1. **Data**: synthetic OHLCV series generated with a drift + noise random
   walk and volatility clustering (no external data feed needed).
2. **Features**: 1-day and 5-day returns, 5/20-day moving average ratio,
   10-day rolling volatility, 14-day RSI, volume change, volume vs its
   moving average, and high-low spread.
3. **Labeling**: 1 if next day's close is higher than today's, else 0.
4. **Split**: time-based 80/20 (train on the past, test on the future —
   no shuffling, to avoid lookahead leakage).
5. **Model**: `RandomForestClassifier` (scikit-learn).
6. **Evaluation**: accuracy, precision/recall, confusion matrix, feature
   importances, and a saved chart (`results.png`).

## Run it

```bash
pip install scikit-learn pandas numpy matplotlib
python main.py
```

## Ideas to extend

- Swap the synthetic data generator for real historical data (e.g. via
  a CSV export from NSE/BSE, or `yfinance` if you have internet access
  in your environment).
- Predict magnitude of return instead of just direction (regression).
- Add lagged features (yesterday's RSI, 2-day-ago volume, etc.).
- Try `XGBoost`/`LightGBM` instead of RandomForest.
- Backtest a simple trading rule based on the model's predictions,
  including transaction costs — this is usually where "accuracy" and
  "actually profitable" diverge sharply.
- Add walk-forward validation (retrain on a rolling window) instead of
  one static train/test split.
