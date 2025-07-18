# -*- coding: utf-8 -*-

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

try:
    # Get stock ticker from user
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").upper()

    # Fetch historical data (last 1000 days)
    data = yf.download(ticker, period="1000d")

    # Validate data
    if data.empty or 'Close' not in data.columns:
        raise ValueError("No valid stock data found. Please check the ticker symbol.")

    # Feature engineering
    data['lag_1'] = data['Close'].shift(1)
    data['ma_5'] = data['Close'].rolling(window=5).mean()
    data['ma_10'] = data['Close'].rolling(window=10).mean()
    data['ma_20'] = data['Close'].rolling(window=20).mean()
    data['returns'] = data['Close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=10).std()
    data['intraday_change'] = data['Close'] - data['Open']
    data['volume'] = data['Volume']
    data['ema_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['ema_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    data['bollinger_mid'] = data['Close'].rolling(window=20).mean()
    data['bollinger_std'] = data['Close'].rolling(window=20).std()
    data['bollinger_upper'] = data['bollinger_mid'] + (2 * data['bollinger_std'])
    data['bollinger_lower'] = data['bollinger_mid'] - (2 * data['bollinger_std'])
    # MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    # Cumulative return
    data['cum_return'] = (1 + data['returns']).cumprod()
    data = data.dropna()

    # Split features and target
    X = data[['lag_1', 'ma_5', 'ma_10', 'ma_20', 'returns', 'volatility', 
          'intraday_change', 'volume', 'ema_5', 'ema_10', 'rsi_14',
          'bollinger_mid', 'bollinger_upper', 'bollinger_lower',
          'macd', 'macd_signal', 'cum_return']]
    y = data['Close']

    #  Train-test split (80%-20%), preserving time order
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #  Train models
    lr = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=2.0).fit(X_train, y_train)
    lasso = Lasso(alpha=0.1).fit(X_train, y_train)
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)
    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)


    # Predict on test set
    y_pred_lr = lr.predict(X_test)
    y_pred_ridge = ridge.predict(X_test)
    y_pred_lasso = lasso.predict(X_test)
    y_pred_enet = elastic_net.predict(X_test)
    y_pred_dt = dt.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Evaluate models
    print(f"Linear Regression R^2: {r2_score(y_test, y_pred_lr):.4f}")
    print(f"Ridge Regression R^2: {r2_score(y_test, y_pred_ridge):.4f}")
    print(f"Lasso Regression R^2: {r2_score(y_test, y_pred_lasso):.4f}")
    print(f"Elastic Net R^2: {r2_score(y_test, y_pred_enet):.4f}")
    print(f"Decision Tree R^2: {r2_score(y_test, y_pred_dt):.4f}")
    print(f"Random Forest R^2: {r2_score(y_test, y_pred_rf):.4f}")


    # Predict next day closing price using Ridge (best regularized model)
    latest = data.iloc[-1:][['lag_1','ma_5', 'ma_10',  'ma_20', 'returns', 'volatility', 
          'intraday_change', 'volume', 'ema_5', 'ema_10', 'rsi_14',
          'bollinger_mid', 'bollinger_upper', 'bollinger_lower',
          'macd', 'macd_signal', 'cum_return']]
    next_day_pred1 = ridge.predict(latest)
    next_day_pred2 = lasso.predict(latest)
    next_day_pred3 = elastic_net.predict(latest)
    next_day_pred4 = dt.predict(latest)
    next_day_pred5 = rf.predict(latest)
    print(f"📌 Predicted NEXT closing price (Ridge): ${next_day_pred1.item():.2f}")
    print(f"📌 Predicted NEXT closing price (Lasso): ${next_day_pred2.item():.2f}")
    print(f"📌 Predicted NEXT closing price (Elastic Net): ${next_day_pred3.item():.2f}")
    print(f"📌 Predicted NEXT closing price (Ridge): ${next_day_pred4.item():.2f}")
    print(f"📌 Predicted NEXT closing price (Ridge): ${next_day_pred5.item():.2f}")

    # Plot actual vs predicted for Ridge and Linear models
    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label='Actual', linewidth=2)
    plt.plot(y_test.index, y_pred_lr, label='Linear Predicted', linestyle='--')
    plt.plot(y_test.index, y_pred_ridge, label='Ridge Predicted', linestyle=':')
    plt.plot(y_test.index, y_pred_lasso, label='Lasso Predicted', linestyle='-.')
    plt.plot(y_test.index, y_pred_enet, label='Elastic Net Predicted', linestyle='-')
    plt.plot(y_test.index, y_pred_dt, label='Decision Tree Predicted', linestyle='--')
    plt.plot(y_test.index, y_pred_rf, label='Random Forest Predicted', linestyle=':')
    plt.legend()
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"❌ Error: {e}")
    print("Please enter a valid stock ticker like AAPL, MSFT, or TSLA.")
