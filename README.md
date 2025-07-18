# 📈 Stock Price Prediction with Decision Trees and Random Forest

This project builds a stock price prediction tool using historical data fetched from Yahoo Finance via the `yfinance` API. It leverages feature engineering and machine learning models (Decision Tree and Random Forest Regressors) to predict the next closing price of a stock.

---

## 🚀 Features

- Uses **Yahoo Finance API** to fetch stock data
- Extracts technical indicators:
  - Moving Averages (MA)
  - Exponential Moving Averages (EMA)
  - Bollinger Bands
  - RSI (Relative Strength Index)
  - MACD and Signal Line
  - Volatility and returns
- Outputs:
  - R² scores for both models
  - Predicted next-day closing prices
- Visualizes actual vs predicted stock prices (optional extension)

---

## 📦 Requirements

Make sure you have Python 3.8+ and install the following packages:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn
